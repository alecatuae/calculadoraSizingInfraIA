#!/usr/bin/env python3
"""
Sizing de Infraestrutura para Inferência de LLMs em GPUs NVIDIA.

Entrypoint principal do sistema modular.
"""

import sys
from typing import Dict, List

from sizing.cli import parse_cli_args
from sizing.config_loader import ConfigLoader
from sizing.capacity_policy import load_capacity_policy
from sizing.platform_storage import load_platform_storage_profile
from sizing.calc_kv import calc_kv_cache
from sizing.calc_vram import calc_vram
from sizing.calc_scenarios import create_scenario_configs, calc_scenario, ScenarioResult
from sizing.calc_physical import calc_physical_consumption
from sizing.calc_storage import calc_storage_requirements
from sizing.calc_storage_validation import validate_storage_profile, format_validation_report, validation_to_dict
from sizing.calc_warmup import calc_warmup_estimate, format_warmup_report, warmup_to_dict
from sizing.validator import validate_all_configs, print_validation_report
from sizing.report_full import format_full_report, format_json_report
from sizing.report_exec import format_exec_summary, format_executive_markdown
from sizing.writer import ReportWriter
from sizing.calc_response_time import (
    calc_latency_analysis, has_performance_data, load_latency_benchmarks
)


def main():
    """Função principal: orquestra todo o fluxo de sizing."""
    
    try:
        # 1. Parse CLI
        config = parse_cli_args()
        
        # 2. Se --validate-only, executar apenas validação
        if config.validate_only:
            print("\n" + "="*100)
            print("MODO DE VALIDAÇÃO: Validando schemas e constraints")
            print("="*100 + "\n")
            
            loader = ConfigLoader(base_path=".", validate=True)
            
            # Carregar todos os arquivos (isso já valida schemas)
            try:
                loader.load_models()
                loader.load_servers()
                loader.load_storage()
            except ValueError as e:
                print(f"\n{e}\n")
                sys.exit(1)
            
            # Obter dados brutos para validação adicional
            models_data, servers_data, storage_data = loader.get_raw_data()
            
            # Validar tudo
            errors, warnings = validate_all_configs(models_data, servers_data, storage_data)
            
            # Validar consistência física de storage (IOPS/Throughput/BlockSize)
            print("\n" + "="*100)
            print("VALIDAÇÃO DE STORAGE (Consistência Física IOPS/Throughput/BlockSize)")
            print("="*100)
            
            for profile_dict in storage_data:
                storage_profile = loader.get_storage(profile_dict["name"])
                storage_validation = validate_storage_profile(storage_profile)
                print(format_validation_report(storage_validation))
                
                if storage_validation.overall_status == "error":
                    errors.append(f"Storage profile '{storage_profile.name}' tem divergência física crítica (>25%)")
                elif storage_validation.overall_status == "warning":
                    warnings.append(f"Storage profile '{storage_profile.name}' tem divergência física moderada (10-25%)")
            
            # Imprimir relatório final
            success = print_validation_report(errors, warnings)
            
            sys.exit(0 if success else 1)
        
        # 3. Modo normal: carregar configurações
        if config.verbose:
            print("🔧 Carregando configurações...")
        
        # Carregar especificações (com validação automática)
        loader = ConfigLoader(base_path=".", validate=True)
        loader.load_models()
        loader.load_servers()
        loader.load_storage()
        
        model = loader.get_model(config.model_name)
        server = loader.get_server(config.server_name)
        storage = loader.get_storage(config.storage_name)
        
        # Carregar política de capacidade
        capacity_policy = load_capacity_policy(
            filepath="parameters.json",
            override_margin=config.capacity_margin
        )
        
        # Aplicar override de target_load_time se fornecido via CLI
        if config.target_load_time is not None:
            if config.target_load_time <= 0:
                print(f"❌ ERRO: --target-load-time deve ser > 0: {config.target_load_time}")
                sys.exit(1)
            if config.target_load_time < 10:
                print(f"⚠️  AVISO: --target-load-time muito baixo ({config.target_load_time}s). Valores < 10s podem não ser viáveis com storage real.")
            capacity_policy.target_load_time_sec = config.target_load_time
            if config.verbose:
                print(f"   ⚠️  Override: Tempo de carga = {config.target_load_time}s (CLI)")
        
        # Carregar profile de storage da plataforma
        platform_storage_profile = load_platform_storage_profile(
            filepath="platform_storage_profile.json"
        )
        
        # Validar SLOs de latência se especificados
        if config.ttft is not None:
            if config.ttft < 100:
                print("❌ ERRO: --ttft deve ser >= 100ms (latências realistas para LLMs de produção)")
                sys.exit(1)
            if config.ttft > 10000:
                print("❌ ERRO: --ttft deve ser <= 10000ms (10s, limite de experiência aceitável)")
                sys.exit(1)
            if config.ttft_p99 is not None and config.ttft_p99 < config.ttft:
                print("❌ ERRO: --ttft-p99 deve ser >= --ttft")
                sys.exit(1)
            benchmarks = load_latency_benchmarks()
            ttft_excellent = benchmarks.get('ttft_excellent_ms', 500)
            ttft_acceptable = benchmarks.get('ttft_acceptable_ms', 2000)
            if config.ttft < ttft_excellent:
                print(f"ℹ️  TTFT alvo: {config.ttft}ms - Excelente (< {ttft_excellent}ms)")
            elif config.ttft <= ttft_acceptable:
                print(f"ℹ️  TTFT alvo: {config.ttft}ms - Aceitável (padrão da indústria: {ttft_excellent}-{ttft_acceptable}ms)")
            else:
                print(f"⚠️  TTFT alvo: {config.ttft}ms - Lento (> {ttft_acceptable}ms, usuário perceberá demora)")

        if config.tpot is not None:
            if config.tpot < 1.0:
                print("❌ ERRO: --tpot deve ser >= 1.0 tokens/s (mínimo viável)")
                sys.exit(1)
            if config.tpot > 200.0:
                print("❌ ERRO: --tpot deve ser <= 200.0 tokens/s (limite físico realista)")
                sys.exit(1)
            benchmarks = load_latency_benchmarks()
            tpot_excellent = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
            tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)
            if config.tpot > tpot_excellent:
                print(f"ℹ️  TPOT alvo: {config.tpot} tok/s - Excelente (> {tpot_excellent} tok/s)")
            elif config.tpot >= tpot_acceptable:
                print(f"ℹ️  TPOT alvo: {config.tpot} tok/s - Aceitável (padrão da indústria: {tpot_acceptable}-{tpot_excellent} tok/s)")
            else:
                print(f"⚠️  TPOT alvo: {config.tpot} tok/s - Baixo (< {tpot_acceptable} tok/s, streaming pode ser lento)")

        if config.verbose:
            print(f"   ✓ Modelo: {model.name}")
            print(f"   ✓ Servidor: {server.name}")
            print(f"   ✓ Storage: {storage.name}")
            margin_source = "CLI override" if config.capacity_margin is not None else "parameters.json"
            print(f"   ✓ Margem de Capacidade: {capacity_policy.margin_percent*100:.0f}% ({margin_source})")
            load_time_source = "CLI override" if config.target_load_time is not None else "parameters.json"
            print(f"   ✓ Tempo de Carga Alvo: {capacity_policy.target_load_time_sec:.0f}s ({load_time_source})")
            print(f"   ✓ Plataforma Storage: {platform_storage_profile.total_per_server_gb:.0f} GB/servidor ({platform_storage_profile.total_per_server_tb:.2f} TB)")
            if config.ttft:
                print(f"   ✓ SLO TTFT: {config.ttft}ms (P50), {config.ttft_p99 or 'auto'}ms (P99)")
            if config.tpot:
                print(f"   ✓ SLO TPOT: {config.tpot} tok/s")
        
        # 4. Calcular KV cache
        if config.verbose:
            print("📊 Calculando KV cache...")
        
        kv_result = calc_kv_cache(
            model=model,
            effective_context=config.effective_context,
            kv_precision=config.kv_precision,
            concurrency=config.concurrency
        )
        
        # 4. Coletar todos os warnings
        all_warnings: List[str] = []
        all_warnings.extend(kv_result.warnings)
        
        # 5. Determinar weights_precision
        weights_precision = config.weights_precision or model.default_weights_precision or "fp8"
        
        # 6. Calcular VRAM base (para cenário recomendado)
        if config.verbose:
            print("💾 Calculando VRAM...")
        
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
        
        # 6.5. VALIDAR CONSISTÊNCIA FÍSICA DE STORAGE (CRÍTICO - PODE BLOQUEAR)
        if config.verbose:
            print("🔍 Validando consistência física de storage (IOPS/Throughput/BlockSize)...")
        
        storage_validation = validate_storage_profile(storage)
        
        # Exibir validação no stdout
        print("\n" + "=" * 100)
        print("VALIDAÇÃO DE STORAGE")
        print("=" * 100)
        print(format_validation_report(storage_validation))
        print("=" * 100 + "\n")
        
        # BLOQUEIO: Se status == "error", NÃO gerar relatórios
        if storage_validation.overall_status == "error":
            print("\n❌ ERRO CRÍTICO: Divergência física no perfil de storage.")
            print(f"   Profile: {storage.name}")
            print(f"\n   {storage_validation.overall_status.upper()}: Inconsistência entre IOPS, Throughput e Block Size.\n")
            print("   A fórmula física Throughput(MB/s) = (IOPS × BlockSize(KB)) / 1024 não é respeitada.")
            print(f"   Divergência > {25:.0f}% (threshold de erro).\n")
            print("   Corrija o arquivo storage.json com valores fisicamente consistentes.")
            print("   Relatórios NÃO serão gerados.\n")
            sys.exit(1)
        
        # Se status == "warning", adicionar aos alertas mas prosseguir
        if storage_validation.overall_status == "warning":
            all_warnings.extend(storage_validation.messages)
            all_warnings.extend(storage_validation.read_validation.messages)
            all_warnings.extend(storage_validation.write_validation.messages)
        
        # 6.6. CALCULAR WARMUP/COLD START
        if config.verbose:
            print("🔥 Calculando estimativa de warmup/cold start...")
        
        # Determinar tamanho do artefato (usar weights_memory se não especificado)
        artifact_size_gib = config.model_artifact_size_gib
        if artifact_size_gib is None:
            # Usar memória de pesos como proxy
            artifact_size_gib = vram_result.fixed_model_gib
        
        warmup_estimate = calc_warmup_estimate(
            storage=storage,
            artifact_size_gib=artifact_size_gib,
            warmup_concurrency=config.warmup_concurrency,
            read_pattern=config.warmup_read_pattern,
            utilization_ratio=config.warmup_utilization_ratio
        )
        
        # 7. Criar configurações dos 3 cenários
        scenario_configs = create_scenario_configs(
            peak_headroom_ratio=config.peak_headroom_ratio,
            kv_budget_ratio=config.kv_budget_ratio
        )
        
        # 8. Calcular cada cenário
        if config.verbose:
            print("🎯 Calculando cenários (Mínimo, Recomendado, Ideal)...")
        
        scenarios: Dict[str, ScenarioResult] = {}
        storage_warnings: List[str] = []
        
        for key, scenario_config in scenario_configs.items():
            # Para cada cenário, recalcular VRAM com kv_budget_ratio específico
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
            
            # Calcular físico
            calc_physical_consumption(scenario, server)
            
            # Calcular storage
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
                retention_days=30  # Será sobrescrito por cenário em calc_storage
            )
            scenario.storage = storage_reqs
            
            # Calcular físico de storage
            scenario.storage_rack_u = storage.rack_units_u
            scenario.storage_power_kw = storage.power_kw
            
            # Calcular totais (Compute + Storage)
            scenario.total_power_kw_with_storage = scenario.total_power_kw + scenario.storage_power_kw
            scenario.total_rack_u_with_storage = scenario.total_rack_u + scenario.storage_rack_u
            
            # Gerar alertas de storage com impacto e ações recomendadas
            # Calcular métricas auxiliares para alertas mais ricos
            storage_throughput_read_gbps = storage.throughput_read_mbps / 1024.0
            storage_throughput_write_gbps = storage.throughput_write_mbps / 1024.0
            
            # 1. Alerta de Volumetria Excedida (CRÍTICO)
            if storage_reqs.storage_total_recommended_tb > storage.usable_capacity_tb:
                deficit_tb = storage_reqs.storage_total_recommended_tb - storage.usable_capacity_tb
                storage_warnings.append(
                    f"🚨 [{scenario_config.name}] Volumetria total RECOMENDADA ({storage_reqs.storage_total_recommended_tb:.2f} TB, base: {storage_reqs.storage_total_base_tb:.2f} TB) "
                    f"excede capacidade utilizável do storage ({storage.usable_capacity_tb:.2f} TB). "
                    f"IMPACTO: Faltará espaço para subir ou escalar o ambiente (déficit: {deficit_tb:.2f} TB). "
                    f"AÇÃO: Recomenda-se storage com capacidade mínima de {storage_reqs.storage_total_recommended_tb:.2f} TB ou reduzir retenção de logs."
                )
            
            # 2. Alerta de IOPS Leitura Excedido
            if storage_reqs.iops_read_peak > storage.iops_read_max:
                iops_deficit = storage_reqs.iops_read_peak - storage.iops_read_max
                iops_factor = storage_reqs.iops_read_peak / storage.iops_read_max
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] IOPS leitura pico ({storage_reqs.iops_read_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_read_max:,}). "
                    f"IMPACTO: Operações de leitura ficarão {iops_factor:.1f}x mais lentas sob carga pico, causando latência no acesso ao cache. "
                    f"AÇÃO: Upgrade storage com IOPS mínimo de {storage_reqs.iops_read_peak:,} ou reduza a concorrência."
                )
            
            # 3. Alerta de IOPS Escrita Excedido
            if storage_reqs.iops_write_peak > storage.iops_write_max:
                iops_write_deficit = storage_reqs.iops_write_peak - storage.iops_write_max
                iops_write_factor = storage_reqs.iops_write_peak / storage.iops_write_max
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] IOPS escrita pico ({storage_reqs.iops_write_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_write_max:,}). "
                    f"IMPACTO: Flush de logs ficará {iops_write_factor:.1f}x mais lento sob carga pico, podendo causar perda de dados se buffers saturarem. "
                    f"AÇÃO: Upgrade storage com IOPS mínimo de {storage_reqs.iops_write_peak:,} ou reduza o volume de logging."
                )
            
            # 4. Alerta de Throughput Leitura Excedido
            if storage_reqs.throughput_read_peak_gbps > storage_throughput_read_gbps:
                throughput_deficit = storage_reqs.throughput_read_peak_gbps - storage_throughput_read_gbps
                throughput_factor = storage_reqs.throughput_read_peak_gbps / storage_throughput_read_gbps
                actual_load_time = capacity_policy.target_load_time_sec * throughput_factor
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] Throughput leitura pico ({storage_reqs.throughput_read_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_read_gbps:.2f} GB/s). "
                    f"IMPACTO: Tempo de restart/scale-out será {throughput_factor:.1f}x maior (~{actual_load_time:.0f}s ao invés de {capacity_policy.target_load_time_sec:.0f}s), aumentando RTO. "
                    f"AÇÃO: Aumente throughput do storage para {storage_reqs.throughput_read_peak_gbps:.2f} GB/s ou ajuste target_load_time_sec para {actual_load_time:.0f}s em parameters.json."
                )
            
            # 5. Alerta de Throughput Escrita Excedido
            if storage_reqs.throughput_write_peak_gbps > storage_throughput_write_gbps:
                throughput_write_deficit = storage_reqs.throughput_write_peak_gbps - storage_throughput_write_gbps
                throughput_write_factor = storage_reqs.throughput_write_peak_gbps / storage_throughput_write_gbps
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] Throughput escrita pico ({storage_reqs.throughput_write_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage_throughput_write_gbps:.2f} GB/s). "
                    f"IMPACTO: Flush de logs ficará {throughput_write_factor:.1f}x mais lento sob carga pico, podendo causar backpressure e perda de logs. "
                    f"AÇÃO: Upgrade storage com throughput mínimo de {storage_reqs.throughput_write_peak_gbps:.2f} GB/s ou reduza o volume de logging."
                )
            
            # 6. Alerta de Volumetria Mínima Acima de 80%
            if key == "minimum" and storage_reqs.storage_total_recommended_tb / storage.usable_capacity_tb > 0.80:
                usage_pct = storage_reqs.storage_total_recommended_tb / storage.usable_capacity_tb * 100
                headroom_tb = storage.usable_capacity_tb - storage_reqs.storage_total_recommended_tb
                storage_warnings.append(
                    f"⚠️ [MÍNIMO] Volumetria recomendada opera acima de 80% da capacidade utilizável "
                    f"({usage_pct:.1f}%, sobra apenas {headroom_tb:.2f} TB). "
                    f"IMPACTO: Risco operacional elevado - sem margem para crescimento orgânico, picos de logs ou rollback de versões. "
                    f"AÇÃO: Considere storage com capacidade adicional de {storage_reqs.storage_total_recommended_tb * 0.3:.2f} TB (~30% buffer) ou reduza retenção de logs."
                )
            
            # Calcular análise de latência TTFT/TPOT se SLOs definidos
            latency = None
            if config.ttft is not None or config.tpot is not None:
                if not has_performance_data(model, server):
                    if config.verbose:
                        print(f"   ⚠️  Dados de performance não encontrados para {model.name} em {server.gpu.model}. Usando estimativa genérica.")
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

                # Emitir alertas de SLO de latência
                if latency and latency.status == 'SLO_VIOLATION':
                    benchmarks = load_latency_benchmarks()
                    ttft_acceptable = benchmarks.get('ttft_acceptable_ms', 2000)
                    tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)
                    tpot_good = benchmarks.get('tpot_good_tokens_per_sec', 8)
                    tpot_excellent = benchmarks.get('tpot_excellent_tokens_per_sec', 10)

                    alert_lines = [f"⚠️  [{scenario_config.name}] SLO de Latência NÃO ATENDIDO:"]

                    if config.ttft and not latency.ttft_p50_ok:
                        deficit = latency.ttft_p50_ms - config.ttft
                        alert_lines.append(
                            f"   TTFT P50: esperado {latency.ttft_p50_ms:.0f}ms, SLO {config.ttft}ms "
                            f"(déficit: {deficit:.0f}ms, +{abs(latency.ttft_p50_margin_percent):.1f}%). "
                            f"Qualidade: {latency.ttft_quality.upper()}. "
                            f"IMPACTO: Usuário percebe latência {'significativa' if latency.ttft_p50_ms > ttft_acceptable else 'moderada'} antes do primeiro token. "
                            f"AÇÃO: {latency.recommendation.strip()}"
                        )

                    if config.tpot and not latency.tpot_ok:
                        deficit = (config.tpot or 0) - latency.tpot_tokens_per_sec
                        alert_lines.append(
                            f"   TPOT: esperado {latency.tpot_tokens_per_sec:.2f} tok/s, SLO {config.tpot} tok/s "
                            f"(déficit: {deficit:.2f} tok/s, {abs(latency.tpot_margin_percent):.1f}% abaixo). "
                            f"ITL: {latency.itl_ms_per_token:.0f}ms/token. "
                            f"Qualidade: {latency.tpot_quality.upper()}. "
                            f"IMPACTO: Streaming {'impraticável' if latency.tpot_tokens_per_sec < tpot_acceptable else 'lento'} - "
                            f"usuário percebe geração token a token. "
                            f"GARGALO: {latency.bottleneck}"
                        )

                    all_warnings.append('\n   '.join(alert_lines))

                elif latency and latency.status == 'SLO_MARGINAL':
                    all_warnings.append(
                        f"ℹ️  [{scenario_config.name}] SLO de Latência ATENDIDO COM MARGEM MÍNIMA "
                        f"(TTFT P50: {latency.ttft_p50_ms:.0f}ms/{config.ttft or 'N/A'}ms, "
                        f"TPOT: {latency.tpot_tokens_per_sec:.2f}/{config.tpot or 'N/A'} tok/s). "
                        f"Margem < 10% — monitorar em produção."
                    )

            scenarios[key] = scenario
        
        # Consolidar alertas de storage
        all_warnings.extend(storage_warnings)
        
        # 9. Gerar relatórios
        if config.verbose:
            print("📝 Gerando relatórios...")
        
        full_report_text = format_full_report(
            model=model,
            server=server,
            storage=storage,
            scenarios=scenarios,
            concurrency=config.concurrency,
            effective_context=kv_result.effective_context_clamped,
            kv_precision=config.kv_precision,
            warnings=all_warnings
        )
        
        full_report_json = format_json_report(
            model=model,
            server=server,
            storage=storage,
            scenarios=scenarios,
            concurrency=config.concurrency,
            effective_context=kv_result.effective_context_clamped,
            kv_precision=config.kv_precision,
            warnings=all_warnings
        )
        
        # 10. Escrever arquivos
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
        
        # 10.5. Gerar relatório executivo em Markdown se solicitado
        if config.executive_report:
            if config.verbose:
                print("📊 Gerando relatório executivo...")
            
            exec_markdown = format_executive_markdown(
                model=model,
                server=server,
                scenarios=scenarios,
                concurrency=config.concurrency,
                effective_context=kv_result.effective_context_clamped,
                kv_precision=config.kv_precision,
                storage_name=storage.name
            )
            
            exec_path = writer.write_executive_report(
                exec_markdown,
                model.name,
                server.name
            )
        
        # 11. Exibir resumo executivo no terminal
        exec_summary = format_exec_summary(
            model_name=model.name,
            server_name=server.name,
            effective_context=kv_result.effective_context_clamped,
            concurrency=config.concurrency,
            kv_precision=config.kv_precision,
            scenarios=scenarios,
            text_report_path=str(text_path),
            json_report_path=str(json_path)
        )
        
        print(exec_summary)
        
        # 11.5. Exibir path do executive report se foi gerado
        if config.executive_report:
            print(f"   • Executive: {exec_path}")
            print()
        
        # 12. Exibir avisos críticos se houver
        critical_warnings = [w for w in all_warnings if "🚨" in w or "ERRO CRÍTICO" in w]
        if critical_warnings:
            print("\n⚠️  AVISOS CRÍTICOS:")
            for warning in critical_warnings:
                print(f"   {warning}")
            print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Operação cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO: {e}", file=sys.stderr)
        if config.verbose if 'config' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
