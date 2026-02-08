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
from sizing.report_full import format_full_report, format_json_report
from sizing.report_exec import format_exec_summary
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
            
            scenarios[key] = scenario
        
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
