#!/usr/bin/env python3
"""
Sizing de Infraestrutura para Inferência de LLMs em GPUs NVIDIA.

Entrypoint principal do sistema modular.
"""

import sys
from typing import Dict, List

from sizing.cli import parse_cli_args
from sizing.config_loader import ConfigLoader
from sizing.calc_kv import calc_kv_cache
from sizing.calc_vram import calc_vram
from sizing.calc_scenarios import create_scenario_configs, calc_scenario, ScenarioResult
from sizing.calc_physical import calc_physical_consumption
from sizing.calc_storage import calc_storage_requirements
from sizing.report_full import format_full_report, format_json_report
from sizing.report_exec import format_exec_summary, format_executive_markdown
from sizing.writer import ReportWriter


def main():
    """Função principal: orquestra todo o fluxo de sizing."""
    
    try:
        # 1. Parse CLI
        config = parse_cli_args()
        
        if config.verbose:
            print("🔧 Carregando configurações...")
        
        # 2. Carregar especificações
        loader = ConfigLoader()
        loader.load_models()
        loader.load_servers()
        loader.load_storage()
        
        model = loader.get_model(config.model_name)
        server = loader.get_server(config.server_name)
        storage = loader.get_storage(config.storage_name)
        
        if config.verbose:
            print(f"   ✓ Modelo: {model.name}")
            print(f"   ✓ Servidor: {server.name}")
            print(f"   ✓ Storage: {storage.name}")
        
        # 3. Calcular KV cache
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
            
            # Gerar alertas de storage
            if storage_reqs.storage_total_tb > storage.usable_capacity_tb:
                storage_warnings.append(
                    f"🚨 [{scenario_config.name}] Volumetria total ({storage_reqs.storage_total_tb:.2f} TB) "
                    f"excede capacidade utilizável do storage ({storage.usable_capacity_tb:.2f} TB)"
                )
            
            if storage_reqs.iops_read_peak > storage.iops_read_max:
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] IOPS leitura pico ({storage_reqs.iops_read_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_read_max:,})"
                )
            
            if storage_reqs.iops_write_peak > storage.iops_write_max:
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] IOPS escrita pico ({storage_reqs.iops_write_peak:,}) "
                    f"excede capacidade do storage ({storage.iops_write_max:,})"
                )
            
            if storage_reqs.throughput_read_peak_gbps > storage.throughput_read_gbps:
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] Throughput leitura pico ({storage_reqs.throughput_read_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage.throughput_read_gbps:.2f} GB/s)"
                )
            
            if storage_reqs.throughput_write_peak_gbps > storage.throughput_write_gbps:
                storage_warnings.append(
                    f"⚠️ [{scenario_config.name}] Throughput escrita pico ({storage_reqs.throughput_write_peak_gbps:.2f} GB/s) "
                    f"excede capacidade do storage ({storage.throughput_write_gbps:.2f} GB/s)"
                )
            
            # Alerta se cenário mínimo opera próximo do limite (>80%)
            if key == "minimum" and storage_reqs.storage_total_tb / storage.usable_capacity_tb > 0.80:
                storage_warnings.append(
                    f"⚠️ [MÍNIMO] Volumetria opera acima de 80% da capacidade utilizável "
                    f"({storage_reqs.storage_total_tb / storage.usable_capacity_tb * 100:.1f}%). "
                    "Risco operacional elevado."
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
