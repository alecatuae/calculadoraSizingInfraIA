#!/usr/bin/env python3
"""
Sizing de Infraestrutura para Inferência de LLMs.

Entrypoint principal do sistema modular.

MODO A — Sizing por Concorrência: --concurrency
MODO B — Sizing por SLO:          --ttft e --tpot
"""

import sys
from typing import Dict, List

from sizing.cli import parse_cli_args
from sizing.config_loader import ConfigLoader
from sizing.capacity_policy import load_capacity_policy
from sizing.platform_storage import load_platform_storage_profile
from sizing.calc_kv import calc_kv_cache
from sizing.calc_vram import calc_vram
from sizing.calc_scenarios import (
    create_scenario_configs, calc_scenario, ScenarioResult,
)
from sizing.calc_physical import calc_physical_consumption
from sizing.calc_storage import calc_storage_requirements
from sizing.calc_storage_validation import validate_storage_profile, format_validation_report
from sizing.calc_warmup import calc_warmup_estimate
from sizing.validator import validate_all_configs, print_validation_report
from sizing.report_full import format_full_report, format_json_report
from sizing.report_exec import format_exec_summary, format_executive_markdown
from sizing.writer import ReportWriter
from sizing.calc_response_time import (
    calc_latency_analysis, calc_max_concurrency_from_slo,
    has_performance_data, load_latency_benchmarks
)


def main():
    """Função principal: orquestra todo o fluxo de sizing."""

    try:
        # 1. Parse CLI
        config = parse_cli_args()

        # 2. Se --validate-only, executar apenas validação
        if config.validate_only:
            print("\n" + "="*100)
            print("MODO DE VALIDACAO: Validando schemas e constraints")
            print("="*100 + "\n")

            loader = ConfigLoader(base_path=".", validate=True)

            try:
                loader.load_models()
                loader.load_servers()
                loader.load_storage()
            except ValueError as e:
                print(f"\n{e}\n")
                sys.exit(1)

            models_data, servers_data, storage_data = loader.get_raw_data()
            errors, warnings = validate_all_configs(models_data, servers_data, storage_data)

            print("\n" + "="*100)
            print("VALIDACAO DE STORAGE (Consistencia Fisica IOPS/Throughput/BlockSize)")
            print("="*100)

            for profile_dict in storage_data:
                storage_profile = loader.get_storage(profile_dict["name"])
                storage_validation = validate_storage_profile(storage_profile)
                print(format_validation_report(storage_validation))

                if storage_validation.overall_status == "error":
                    errors.append(f"Storage profile '{storage_profile.name}' tem divergencia fisica critica (>25%)")
                elif storage_validation.overall_status == "warning":
                    warnings.append(f"Storage profile '{storage_profile.name}' tem divergencia fisica moderada (10-25%)")

            success = print_validation_report(errors, warnings)
            sys.exit(0 if success else 1)

        # 3. Modo normal: carregar configurações
        if config.verbose:
            print("Carregando configuracoes...")

        loader = ConfigLoader(base_path=".", validate=True)
        loader.load_models()
        loader.load_servers()
        loader.load_storage()

        model = loader.get_model(config.model_name)
        server = loader.get_server(config.server_name)
        storage = loader.get_storage(config.storage_name)

        capacity_policy = load_capacity_policy(
            filepath="parameters.json",
            override_margin=config.capacity_margin
        )

        if config.target_load_time is not None:
            if config.target_load_time <= 0:
                print(f"ERRO: --target-load-time deve ser > 0: {config.target_load_time}")
                sys.exit(1)
            if config.target_load_time < 10:
                print(f"AVISO: --target-load-time muito baixo ({config.target_load_time}s). Valores < 10s podem nao ser viaveis com storage real.")
            capacity_policy.target_load_time_sec = config.target_load_time

        platform_storage_profile = load_platform_storage_profile(
            filepath="platform_storage_profile.json"
        )

        benchmarks = load_latency_benchmarks()
        ttft_excellent = benchmarks.get('ttft_excellent_ms', 500)
        ttft_acceptable = benchmarks.get('ttft_acceptable_ms', 2000)
        tpot_excellent = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
        tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)

        # ── Informar modo de operação ─────────────────────────────────────────
        if config.sizing_mode == "slo_driven":
            print(f"[MODO B — SLO-DRIVEN] TTFT alvo: {config.ttft_input_ms}ms | TPOT alvo: {config.tpot_input_ms} tok/s")

            # Validar range dos SLOs
            if config.ttft_input_ms < 100:
                print("ERRO: --ttft deve ser >= 100ms")
                sys.exit(1)
            if config.ttft_input_ms > 10000:
                print("ERRO: --ttft deve ser <= 10000ms")
                sys.exit(1)
            if config.tpot_input_ms < 1.0:
                print("ERRO: --tpot deve ser >= 1.0 tokens/s")
                sys.exit(1)
            if config.tpot_input_ms > 200.0:
                print("ERRO: --tpot deve ser <= 200.0 tokens/s")
                sys.exit(1)

            if config.ttft_input_ms < ttft_excellent:
                print(f"   [INFO] TTFT alvo {config.ttft_input_ms}ms — Excelente (< {ttft_excellent}ms)")
            elif config.ttft_input_ms <= ttft_acceptable:
                print(f"   [INFO] TTFT alvo {config.ttft_input_ms}ms — Aceitavel (padrao industria)")
            else:
                print(f"   [AVISO] TTFT alvo {config.ttft_input_ms}ms — Lento (> {ttft_acceptable}ms)")

            if config.tpot_input_ms > tpot_excellent:
                print(f"   [INFO] TPOT alvo {config.tpot_input_ms} tok/s — Excelente")
            elif config.tpot_input_ms >= tpot_acceptable:
                print(f"   [INFO] TPOT alvo {config.tpot_input_ms} tok/s — Aceitavel")
            else:
                print(f"   [AVISO] TPOT alvo {config.tpot_input_ms} tok/s — Baixo (streaming pode ser lento)")

            # Calcular P99 derivado
            if config.ttft_p99 is None:
                ttft_p99_multiplier = float(_load_param('ttft_p99_multiplier', 2.0))
                effective_ttft_p99 = int(config.ttft_input_ms * ttft_p99_multiplier)
            else:
                effective_ttft_p99 = config.ttft_p99

        else:
            print(f"[MODO A — CONCORRENCIA-DRIVEN] Sessoes simultaneas: {config.concurrency_input:,}")
            effective_ttft_p99 = None

        if config.verbose:
            print(f"   Modelo: {model.name}")
            print(f"   Servidor de Inferencia: {server.name}")
            print(f"   Storage: {storage.name}")
            margin_source = "CLI override" if config.capacity_margin is not None else "parameters.json"
            print(f"   Margem de Capacidade: {capacity_policy.margin_percent*100:.0f}% ({margin_source})")
            load_time_source = "CLI override" if config.target_load_time is not None else "parameters.json"
            print(f"   Tempo de Carga Alvo: {capacity_policy.target_load_time_sec:.0f}s ({load_time_source})")
            print(f"   Plataforma Storage: {platform_storage_profile.total_per_server_gb:.0f} GB/servidor")

        # 4. Calcular KV cache
        if config.verbose:
            print("Calculando KV cache...")

        kv_result = calc_kv_cache(
            model=model,
            effective_context=config.effective_context,
            kv_precision=config.kv_precision,
            concurrency=config.concurrency
        )

        all_warnings: List[str] = []
        all_warnings.extend(kv_result.warnings)

        weights_precision = config.weights_precision or model.default_weights_precision or "fp8"

        if config.verbose:
            print("Calculando VRAM...")

        vram_result = calc_vram(
            model=model,
            server=server,
            kv_gib_per_session=kv_result.kv_gib_per_session,
            concurrency=config.concurrency,
            runtime_overhead_gib=config.runtime_overhead_gib,
            kv_budget_ratio=config.kv_budget_ratio,
            weights_precision=weights_precision,
            weights_memory_override=config.weights_memory_gib,
            replicas_per_node=config.replicas_per_node,
            tensor_parallel=config.tensor_parallel,
            pipeline_parallel=config.pipeline_parallel
        )

        all_warnings.extend(vram_result.warnings)

        if config.verbose:
            print("Validando consistencia fisica de storage (IOPS/Throughput/BlockSize)...")

        storage_validation = validate_storage_profile(storage)

        print("\n" + "=" * 100)
        print("VALIDACAO DE STORAGE")
        print("=" * 100)
        print(format_validation_report(storage_validation))
        print("=" * 100 + "\n")

        if storage_validation.overall_status == "error":
            print("\nERRO CRITICO: Divergencia fisica no perfil de storage.")
            print(f"   Profile: {storage.name}")
            print(f"\n   {storage_validation.overall_status.upper()}: Inconsistencia entre IOPS, Throughput e Block Size.\n")
            print("   A formula fisica Throughput(MB/s) = (IOPS x BlockSize(KB)) / 1024 nao e respeitada.")
            print(f"   Divergencia > {25:.0f}% (threshold de erro).\n")
            print("   Corrija o arquivo storage.json com valores fisicamente consistentes.")
            print("   Relatorios NAO serao gerados.\n")
            sys.exit(1)

        if storage_validation.overall_status == "warning":
            all_warnings.extend(storage_validation.messages)
            all_warnings.extend(storage_validation.read_validation.messages)
            all_warnings.extend(storage_validation.write_validation.messages)

        if config.verbose:
            print("Calculando estimativa de warmup/cold start...")

        artifact_size_gib = config.model_artifact_size_gib
        if artifact_size_gib is None:
            artifact_size_gib = vram_result.fixed_model_gib

        warmup_estimate = calc_warmup_estimate(
            storage=storage,
            artifact_size_gib=artifact_size_gib,
            warmup_concurrency=config.warmup_concurrency,
            read_pattern=config.warmup_read_pattern,
            utilization_ratio=config.warmup_utilization_ratio
        )

        scenario_configs = create_scenario_configs(
            peak_headroom_ratio=config.peak_headroom_ratio,
            kv_budget_ratio=config.kv_budget_ratio
        )

        if config.verbose:
            print("Calculando cenarios (Minimo, Recomendado, Ideal)...")

        scenarios: Dict[str, ScenarioResult] = {}
        storage_warnings: List[str] = []

        for key, scenario_config in scenario_configs.items():
            vram_scenario = calc_vram(
                model=model,
                server=server,
                kv_gib_per_session=kv_result.kv_gib_per_session,
                concurrency=config.concurrency,
                runtime_overhead_gib=config.runtime_overhead_gib,
                kv_budget_ratio=scenario_config.kv_budget_ratio,
                weights_precision=weights_precision,
                weights_memory_override=config.weights_memory_gib,
                replicas_per_node=config.replicas_per_node,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel
            )

            scenario = calc_scenario(
                config=scenario_config,
                vram=vram_scenario,
                concurrency=config.concurrency,
                runtime_overhead_gib=config.runtime_overhead_gib
            )

            calc_physical_consumption(scenario, server)

            storage_reqs = calc_storage_requirements(
                model=model,
                server=server,
                storage=storage,
                concurrency=config.concurrency,
                num_nodes=scenario.nodes_final,
                sessions_per_node=vram_scenario.sessions_per_node,
                weights_precision=weights_precision,
                replicas_per_node=config.replicas_per_node,
                capacity_policy=capacity_policy,
                platform_storage_profile=platform_storage_profile,
                scenario=key,
                retention_days=30
            )
            scenario.storage = storage_reqs

            scenario.storage_rack_u = storage.rack_units_u
            scenario.storage_power_kw = storage.power_kw
            scenario.total_power_kw_with_storage = scenario.total_power_kw + scenario.storage_power_kw
            scenario.total_rack_u_with_storage = scenario.total_rack_u + scenario.storage_rack_u

            storage_throughput_read_gbps = storage.throughput_read_mbps / 1024.0
            storage_throughput_write_gbps = storage.throughput_write_mbps / 1024.0

            if storage_reqs.storage_total_recommended_tb > storage.usable_capacity_tb:
                deficit_tb = storage_reqs.storage_total_recommended_tb - storage.usable_capacity_tb
                storage_warnings.append(
                    f"[CRITICO] [{scenario_config.name}] Volumetria total RECOMENDADA ({storage_reqs.storage_total_recommended_tb:.2f} TB) "
                    f"excede capacidade utilizavel do storage ({storage.usable_capacity_tb:.2f} TB). "
                    f"Deficit: {deficit_tb:.2f} TB. "
                    f"Requer storage com capacidade minima de {storage_reqs.storage_total_recommended_tb:.2f} TB."
                )

            if storage_reqs.iops_read_peak > storage.iops_read_max:
                iops_factor = storage_reqs.iops_read_peak / storage.iops_read_max
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] IOPS leitura pico ({storage_reqs.iops_read_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_read_max:,}). "
                    f"Fator: {iops_factor:.1f}x. Storage minimo requerido: {storage_reqs.iops_read_peak:,} IOPS leitura."
                )

            if storage_reqs.iops_write_peak > storage.iops_write_max:
                iops_write_factor = storage_reqs.iops_write_peak / storage.iops_write_max
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] IOPS escrita pico ({storage_reqs.iops_write_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_write_max:,}). "
                    f"Fator: {iops_write_factor:.1f}x."
                )

            if storage_reqs.throughput_read_peak_gbps > storage_throughput_read_gbps:
                throughput_factor = storage_reqs.throughput_read_peak_gbps / storage_throughput_read_gbps
                actual_load_time = capacity_policy.target_load_time_sec * throughput_factor
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] Throughput leitura pico ({storage_reqs.throughput_read_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_read_gbps:.2f} GB/s). "
                    f"Tempo de restart estimado: {actual_load_time:.0f}s (alvo: {capacity_policy.target_load_time_sec:.0f}s). "
                    f"Storage com throughput minimo de {storage_reqs.throughput_read_peak_gbps:.2f} GB/s requerido."
                )

            if storage_reqs.throughput_write_peak_gbps > storage_throughput_write_gbps:
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] Throughput escrita pico ({storage_reqs.throughput_write_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_write_gbps:.2f} GB/s)."
                )

            if not has_performance_data(model, server) and config.verbose:
                print(f"   Dados de performance nao encontrados para {model.name} em {server.gpu.model}. Usando estimativa generica.")

            # ── Capacidade máxima por SLO (Modo B — calculado ANTES da latência) ──
            if config.sizing_mode == "slo_driven":
                slo_cap = calc_max_concurrency_from_slo(
                    model=model,
                    server=server,
                    num_nodes=scenario.nodes_final,
                    sessions_per_node=vram_scenario.sessions_per_node,
                    target_ttft_p50_ms=config.ttft_input_ms,
                    target_tpot_min_tokens_per_sec=config.tpot_input_ms,
                    effective_context=kv_result.effective_context_clamped
                )
                scenario.slo_capacity = slo_cap
                # Usa concorrência derivada dos SLOs para a análise de latência
                latency_concurrency = slo_cap.max_concurrency_combined if slo_cap.is_feasible and slo_cap.max_concurrency_combined > 0 else config.concurrency
            else:
                slo_cap = None
                latency_concurrency = config.concurrency

            # ── Análise de latência TTFT/TPOT (todos os modos) ────────────────
            # Modo A: targets=None → estimativa sem SLO (status=NO_SLO)
            # Modo B: targets definidos → validação contra SLO (usa conc. dos SLOs)
            target_ttft = config.ttft_input_ms if config.sizing_mode == "slo_driven" else None
            target_tpot = config.tpot_input_ms if config.sizing_mode == "slo_driven" else None

            latency = calc_latency_analysis(
                model=model,
                server=server,
                num_nodes=scenario.nodes_final,
                sessions_per_node=scenario.sessions_per_node_effective,
                concurrency=latency_concurrency,
                target_ttft_p50_ms=target_ttft,
                target_ttft_p99_ms=effective_ttft_p99 if config.sizing_mode == "slo_driven" else None,
                target_tpot_min_tokens_per_sec=target_tpot,
                effective_context=kv_result.effective_context_clamped
            )
            scenario.latency = latency

            scenarios[key] = scenario

        # ── Modo B: verificar viabilidade física ──────────────────────────────
        if config.sizing_mode == "slo_driven":
            infeasible_scenarios = [
                k for k in scenarios
                if scenarios[k].slo_capacity and not scenarios[k].slo_capacity.is_feasible
            ]
            if infeasible_scenarios:
                # Se mesmo o ideal for inviável, é erro fatal
                print("\n" + "=" * 80)
                print("ERRO: SLOs FISICAMENTE INVIÁVEIS")
                print("=" * 80)
                for k in infeasible_scenarios:
                    sc = scenarios[k].slo_capacity
                    print(f"\n[{scenarios[k].config.name}] {sc.infeasibility_reason}")
                print(
                    "\nImpossível atender TTFT/TPOT com o modelo e servidor selecionados "
                    "sob as premissas atuais.\n"
                    "Ações possíveis:\n"
                    "  - Reduza o contexto efetivo (--effective-context)\n"
                    "  - Use um servidor de inferência com maior throughput de GPU\n"
                    "  - Ajuste a quantização do modelo\n"
                    "  - Aumente o SLO de TTFT (valor mais permissivo)\n"
                )
                print("Relatórios NÃO serão gerados.")
                sys.exit(1)

        all_warnings.extend(storage_warnings)

        if config.verbose:
            print("Gerando relatorios...")

        full_report_text = format_full_report(
            model=model,
            server=server,
            storage=storage,
            scenarios=scenarios,
            concurrency=config.concurrency,
            effective_context=kv_result.effective_context_clamped,
            kv_precision=config.kv_precision,
            warnings=all_warnings,
            sizing_mode=config.sizing_mode,
            ttft_input_ms=config.ttft_input_ms,
            tpot_input_ms=config.tpot_input_ms,
            concurrency_input=config.concurrency_input
        )

        full_report_json = format_json_report(
            model=model,
            server=server,
            storage=storage,
            scenarios=scenarios,
            concurrency=config.concurrency,
            effective_context=kv_result.effective_context_clamped,
            kv_precision=config.kv_precision,
            warnings=all_warnings,
            sizing_mode=config.sizing_mode
        )

        writer = ReportWriter()

        text_path = writer.write_text_report(
            full_report_text,
            model.name,
            server.name
        )

        json_path = writer.write_json_report(
            full_report_json,
            model.name,
            server.name
        )

        if config.executive_report:
            if config.verbose:
                print("Gerando relatorio executivo...")

            exec_markdown = format_executive_markdown(
                model=model,
                server=server,
                scenarios=scenarios,
                concurrency=config.concurrency,
                effective_context=kv_result.effective_context_clamped,
                kv_precision=config.kv_precision,
                storage_name=storage.name,
                sizing_mode=config.sizing_mode,
                ttft_input_ms=config.ttft_input_ms,
                tpot_input_ms=config.tpot_input_ms,
                concurrency_input=config.concurrency_input
            )

            exec_path = writer.write_executive_report(
                exec_markdown,
                model.name,
                server.name
            )

        exec_summary = format_exec_summary(
            model_name=model.name,
            server_name=server.name,
            effective_context=kv_result.effective_context_clamped,
            concurrency=config.concurrency,
            kv_precision=config.kv_precision,
            scenarios=scenarios,
            text_report_path=str(text_path),
            json_report_path=str(json_path),
            sizing_mode=config.sizing_mode,
            ttft_input_ms=config.ttft_input_ms,
            tpot_input_ms=config.tpot_input_ms,
            concurrency_input=config.concurrency_input
        )

        print(exec_summary)

        if config.executive_report:
            print(f"   Executive: {exec_path}")
            print()

        # Exibir alertas críticos de storage
        critical_warnings = [w for w in all_warnings if "[CRITICO]" in w or "ERRO CRITICO" in w]
        if critical_warnings:
            print("\nALERTAS CRITICOS DE STORAGE:")
            for warning in critical_warnings:
                print(f"   {warning}")
            print()

    except KeyboardInterrupt:
        print("\n\nOperacao cancelada pelo usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        if 'config' in locals() and config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _load_param(param_name: str, default):
    """Carrega parâmetro de parameters.json com fallback."""
    import json
    try:
        with open('parameters.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
            return params.get(param_name, default)
    except Exception:
        return default


if __name__ == "__main__":
    main()
