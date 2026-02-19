#!/usr/bin/env python3
"""
Sizing de Infraestrutura para Inferência de LLMs em GPUs NVIDIA.

Entrypoint principal do sistema modular.
"""

import math
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
    CalibrationRecommendation
)
from sizing.calc_physical import calc_physical_consumption
from sizing.calc_storage import calc_storage_requirements
from sizing.calc_storage_validation import validate_storage_profile, format_validation_report, validation_to_dict
from sizing.calc_warmup import calc_warmup_estimate, format_warmup_report, warmup_to_dict
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
            if config.verbose:
                print(f"   Override: Tempo de carga = {config.target_load_time}s (CLI)")
        
        platform_storage_profile = load_platform_storage_profile(
            filepath="platform_storage_profile.json"
        )

        # Carregar benchmarks para uso em validações e mensagens
        benchmarks = load_latency_benchmarks()
        ttft_excellent = benchmarks.get('ttft_excellent_ms', 500)
        ttft_acceptable = benchmarks.get('ttft_acceptable_ms', 2000)
        tpot_excellent = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
        tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)

        # Informar modo de operação
        if config.sizing_mode == "slo_driven":
            ttft_str = f"TTFT={config.ttft}ms" if config.ttft else ""
            tpot_str = f"TPOT={config.tpot} tok/s" if config.tpot else ""
            slo_desc = " / ".join(filter(None, [ttft_str, tpot_str]))
            print(f"[MODO SLO-DRIVEN] Dimensionamento guiado por {slo_desc}")
            if config.ttft is None or config.tpot is None:
                # Apenas um SLO definido
                print(f"   Concorrencia {config.concurrency} usada para storage/fisico.")
        else:
            # Modo Concorrência-Driven: aplicar SLOs implícitos de parameters.json
            ttft_default = benchmarks.get('ttft_acceptable_ms', 2000)
            tpot_default = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)
            # Aplicar SLOs implícitos se não especificados
            if config.ttft is None:
                config.ttft = ttft_default
            if config.tpot is None:
                config.tpot = float(tpot_default)
            print(f"[MODO CONCORRENCIA-DRIVEN] Latencia calculada com benchmarks padrao: TTFT={config.ttft}ms / TPOT={config.tpot} tok/s (parameters.json)")
            print(f"   Para SLOs personalizados, use --ttft e --tpot.")

        # Validar SLOs de latência se especificados
        if config.ttft is not None:
            if config.ttft < 100:
                print("ERRO: --ttft deve ser >= 100ms (latencias realistas para LLMs de producao)")
                sys.exit(1)
            if config.ttft > 10000:
                print("ERRO: --ttft deve ser <= 10000ms (10s, limite de experiencia aceitavel)")
                sys.exit(1)
            if config.ttft_p99 is not None and config.ttft_p99 < config.ttft:
                print("ERRO: --ttft-p99 deve ser >= --ttft")
                sys.exit(1)
            if config.ttft < ttft_excellent:
                print(f"[INFO] TTFT alvo: {config.ttft}ms - Excelente (< {ttft_excellent}ms)")
            elif config.ttft <= ttft_acceptable:
                print(f"[INFO] TTFT alvo: {config.ttft}ms - Aceitavel (padrao da industria: {ttft_excellent}-{ttft_acceptable}ms)")
            else:
                print(f"[AVISO] TTFT alvo: {config.ttft}ms - Lento (> {ttft_acceptable}ms, usuario percebera demora)")

        if config.tpot is not None:
            if config.tpot < 1.0:
                print("ERRO: --tpot deve ser >= 1.0 tokens/s (minimo viavel)")
                sys.exit(1)
            if config.tpot > 200.0:
                print("ERRO: --tpot deve ser <= 200.0 tokens/s (limite fisico realista)")
                sys.exit(1)
            if config.tpot > tpot_excellent:
                print(f"[INFO] TPOT alvo: {config.tpot} tok/s - Excelente (> {tpot_excellent} tok/s)")
            elif config.tpot >= tpot_acceptable:
                print(f"[INFO] TPOT alvo: {config.tpot} tok/s - Aceitavel (padrao da industria: {tpot_acceptable}-{tpot_excellent} tok/s)")
            else:
                print(f"[AVISO] TPOT alvo: {config.tpot} tok/s - Baixo (< {tpot_acceptable} tok/s, streaming pode ser lento)")

        if config.verbose:
            print(f"   Modelo: {model.name}")
            print(f"   Servidor: {server.name}")
            print(f"   Storage: {storage.name}")
            margin_source = "CLI override" if config.capacity_margin is not None else "parameters.json"
            print(f"   Margem de Capacidade: {capacity_policy.margin_percent*100:.0f}% ({margin_source})")
            load_time_source = "CLI override" if config.target_load_time is not None else "parameters.json"
            print(f"   Tempo de Carga Alvo: {capacity_policy.target_load_time_sec:.0f}s ({load_time_source})")
            print(f"   Plataforma Storage: {platform_storage_profile.total_per_server_gb:.0f} GB/servidor ({platform_storage_profile.total_per_server_tb:.2f} TB)")
            if config.ttft:
                print(f"   SLO TTFT: {config.ttft}ms (P50), {config.ttft_p99 or 'auto'}ms (P99)")
            if config.tpot:
                print(f"   SLO TPOT: {config.tpot} tok/s")
        
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
                    f"[CRITICO] [{scenario_config.name}] Volumetria total RECOMENDADA ({storage_reqs.storage_total_recommended_tb:.2f} TB, base: {storage_reqs.storage_total_base_tb:.2f} TB) "
                    f"excede capacidade utilizavel do storage ({storage.usable_capacity_tb:.2f} TB). "
                    f"IMPACTO: Faltara espaco para subir ou escalar o ambiente (deficit: {deficit_tb:.2f} TB). "
                    f"ACAO: Recomenda-se storage com capacidade minima de {storage_reqs.storage_total_recommended_tb:.2f} TB ou reduzir retencao de logs."
                )
            
            if storage_reqs.iops_read_peak > storage.iops_read_max:
                iops_factor = storage_reqs.iops_read_peak / storage.iops_read_max
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] IOPS leitura pico ({storage_reqs.iops_read_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_read_max:,}). "
                    f"IMPACTO: Operacoes de leitura ficarao {iops_factor:.1f}x mais lentas sob carga pico. "
                    f"ACAO: Upgrade storage com IOPS minimo de {storage_reqs.iops_read_peak:,} ou reduza a concorrencia."
                )
            
            if storage_reqs.iops_write_peak > storage.iops_write_max:
                iops_write_factor = storage_reqs.iops_write_peak / storage.iops_write_max
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] IOPS escrita pico ({storage_reqs.iops_write_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_write_max:,}). "
                    f"IMPACTO: Flush de logs ficara {iops_write_factor:.1f}x mais lento sob carga pico. "
                    f"ACAO: Upgrade storage com IOPS minimo de {storage_reqs.iops_write_peak:,} ou reduza o volume de logging."
                )
            
            if storage_reqs.throughput_read_peak_gbps > storage_throughput_read_gbps:
                throughput_factor = storage_reqs.throughput_read_peak_gbps / storage_throughput_read_gbps
                actual_load_time = capacity_policy.target_load_time_sec * throughput_factor
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] Throughput leitura pico ({storage_reqs.throughput_read_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_read_gbps:.2f} GB/s). "
                    f"IMPACTO: Tempo de restart/scale-out sera {throughput_factor:.1f}x maior (~{actual_load_time:.0f}s ao inves de {capacity_policy.target_load_time_sec:.0f}s), aumentando RTO. "
                    f"ACAO: Aumente throughput do storage para {storage_reqs.throughput_read_peak_gbps:.2f} GB/s ou ajuste target_load_time_sec para {actual_load_time:.0f}s em parameters.json."
                )
            
            if storage_reqs.throughput_write_peak_gbps > storage_throughput_write_gbps:
                throughput_write_factor = storage_reqs.throughput_write_peak_gbps / storage_throughput_write_gbps
                storage_warnings.append(
                    f"[AVISO] [{scenario_config.name}] Throughput escrita pico ({storage_reqs.throughput_write_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_write_gbps:.2f} GB/s). "
                    f"IMPACTO: Flush de logs ficara {throughput_write_factor:.1f}x mais lento sob carga pico. "
                    f"ACAO: Upgrade storage com throughput minimo de {storage_reqs.throughput_write_peak_gbps:.2f} GB/s ou reduza o volume de logging."
                )
            
            if key == "minimum" and storage_reqs.storage_total_recommended_tb / storage.usable_capacity_tb > 0.80:
                usage_pct = storage_reqs.storage_total_recommended_tb / storage.usable_capacity_tb * 100
                headroom_tb = storage.usable_capacity_tb - storage_reqs.storage_total_recommended_tb
                storage_warnings.append(
                    f"[AVISO] [MINIMO] Volumetria recomendada opera acima de 80% da capacidade utilizavel "
                    f"({usage_pct:.1f}%, sobra apenas {headroom_tb:.2f} TB). "
                    f"IMPACTO: Risco operacional elevado - sem margem para crescimento organico, picos de logs ou rollback. "
                    f"ACAO: Considere storage com capacidade adicional de {storage_reqs.storage_total_recommended_tb * 0.3:.2f} TB (~30% buffer) ou reduza retencao de logs."
                )
            
            # Calcular análise de latência TTFT/TPOT
            latency = None
            if config.ttft is not None or config.tpot is not None:
                if not has_performance_data(model, server):
                    if config.verbose:
                        print(f"   Dados de performance nao encontrados para {model.name} em {server.gpu.model}. Usando estimativa generica.")
                latency = calc_latency_analysis(
                    model=model,
                    server=server,
                    num_nodes=scenario.nodes_final,
                    sessions_per_node=scenario.sessions_per_node_effective,
                    concurrency=config.concurrency,
                    target_ttft_p50_ms=config.ttft,
                    target_ttft_p99_ms=config.ttft_p99,
                    target_tpot_min_tokens_per_sec=config.tpot,
                    effective_context=kv_result.effective_context_clamped
                )
                scenario.latency = latency

                # Alertas de SLO de latência
                if latency and latency.status == 'SLO_VIOLATION':
                    alert_lines = [f"[{scenario_config.name}] SLO de Latencia NAO ATENDIDO:"]

                    if config.ttft and not latency.ttft_p50_ok:
                        deficit = latency.ttft_p50_ms - config.ttft
                        alert_lines.append(
                            f"   TTFT P50: esperado {latency.ttft_p50_ms:.0f}ms, SLO {config.ttft}ms "
                            f"(deficit: {deficit:.0f}ms, +{abs(latency.ttft_p50_margin_percent):.1f}%). "
                            f"Qualidade: {latency.ttft_quality.upper()}. "
                            f"IMPACTO: Usuario percebe latencia {'significativa' if latency.ttft_p50_ms > ttft_acceptable else 'moderada'} antes do primeiro token. "
                            f"ACAO: {latency.recommendation.strip()}"
                        )

                    if config.tpot and not latency.tpot_ok:
                        deficit = (config.tpot or 0) - latency.tpot_tokens_per_sec
                        alert_lines.append(
                            f"   TPOT: esperado {latency.tpot_tokens_per_sec:.2f} tok/s, SLO {config.tpot} tok/s "
                            f"(deficit: {deficit:.2f} tok/s, {abs(latency.tpot_margin_percent):.1f}% abaixo). "
                            f"ITL: {latency.itl_ms_per_token:.0f}ms/token. "
                            f"Qualidade: {latency.tpot_quality.upper()}. "
                            f"IMPACTO: Streaming {'impraticavel' if latency.tpot_tokens_per_sec < tpot_acceptable else 'lento'}. "
                            f"GARGALO: {latency.bottleneck}"
                        )

                    all_warnings.append('\n   '.join(alert_lines))

                elif latency and latency.status == 'SLO_MARGINAL':
                    all_warnings.append(
                        f"[{scenario_config.name}] SLO de Latencia ATENDIDO COM MARGEM MINIMA "
                        f"(TTFT P50: {latency.ttft_p50_ms:.0f}ms/{config.ttft or 'N/A'}ms, "
                        f"TPOT: {latency.tpot_tokens_per_sec:.2f}/{config.tpot or 'N/A'} tok/s). "
                        f"Margem < 10% - monitorar em producao."
                    )

            # Calcular capacidade máxima por SLO para todos os cenários
            if config.ttft is not None or config.tpot is not None:
                slo_cap = calc_max_concurrency_from_slo(
                    model=model,
                    server=server,
                    num_nodes=scenario.nodes_final,
                    sessions_per_node=vram_scenario.sessions_per_node,
                    target_ttft_p50_ms=config.ttft,
                    target_tpot_min_tokens_per_sec=config.tpot,
                    effective_context=kv_result.effective_context_clamped
                )
                scenario.slo_capacity = slo_cap

                if not slo_cap.is_feasible:
                    all_warnings.append(
                        f"[INVIAVEL] [{scenario_config.name}] {slo_cap.infeasibility_reason}"
                    )
                elif config.sizing_mode == "slo_driven":
                    print(f"   [{scenario_config.name.upper()}] Concorrencia maxima (SLO-driven): {slo_cap.max_concurrency_combined} sessoes "
                          f"| Gargalo: {slo_cap.limiting_factor} | Util. max: {slo_cap.util_max_from_ttft*100:.1f}%")

                # Calibração para Modo Concorrência-Driven com violação
                if config.sizing_mode == "concurrency_driven" and latency and latency.status == "SLO_VIOLATION":
                    if slo_cap.max_concurrency_combined > 0:
                        nodes_needed = math.ceil(
                            config.concurrency / slo_cap.max_concurrency_combined * scenario.nodes_final
                        )
                    else:
                        nodes_needed = None
                    extra = (nodes_needed - scenario.nodes_final) if nodes_needed else 0
                    scenario.calibration = CalibrationRecommendation(
                        nodes_current=scenario.nodes_final,
                        nodes_recommended=nodes_needed,
                        max_concurrency_current_nodes=slo_cap.max_concurrency_combined,
                        concurrency_requested=config.concurrency,
                        limiting_factor=slo_cap.limiting_factor,
                        extra_nodes_needed=max(0, extra)
                    )
                    print(
                        f"   [CALIBRACAO] [{scenario_config.name.upper()}] Para {config.concurrency} sessoes com SLOs: "
                        f"max. atual = {slo_cap.max_concurrency_combined} | "
                        f"nos recomendados = {nodes_needed or 'N/A'} (+{max(0, extra)} nos)"
                    )

            scenarios[key] = scenario
        
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
            sizing_mode=config.sizing_mode
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
                sizing_mode=config.sizing_mode
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
            sizing_mode=config.sizing_mode
        )
        
        print(exec_summary)
        
        if config.executive_report:
            print(f"   Executive: {exec_path}")
            print()
        
        # Exibir avisos críticos se houver
        critical_warnings = [w for w in all_warnings if "[CRITICO]" in w or "ERRO CRITICO" in w or "[INVIAVEL]" in w]
        if critical_warnings:
            print("\nAVISOS CRITICOS:")
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


if __name__ == "__main__":
    main()
