#!/usr/bin/env python3
"""
Análise Comparativa de Sizing de Modelos LLM

Script para comparar múltiplos relatórios de sizing e identificar o modelo
mais eficiente em diferentes dimensões (KV cache, infraestrutura, custo, VRAM).

Uso:
    python analise_comparativa.py
    python analise_comparativa.py --models "DeepSeek-V3.2,opt-oss-120b"
    python analise_comparativa.py --scenario ideal --format json
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ComparisonMetrics:
    """Métricas extraídas de um relatório para comparação."""
    model: str
    server: str
    concurrency: int
    effective_context: int
    kv_precision: str
    
    # KV Cache Efficiency
    vram_per_session_gib: float
    sessions_per_node_capacity: int
    sessions_per_node_effective: int
    
    # Infrastructure
    nodes_final: int
    hbm_utilization_ratio: float
    
    # VRAM Breakdown
    fixed_model_gib: float
    vram_total_node_gib: float
    vram_model_percent: float
    vram_kv_percent: float
    
    # Physical Resources
    total_power_kw: float
    total_rack_u: int
    storage_total_tb: float
    
    # Efficiency Metrics
    sessions_per_kw: float
    cost_per_session_month: float


def load_sizing_reports(directory: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Carrega todos os JSONs de sizing do diretório com filtros opcionais."""
    reports = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"❌ ERRO: Diretório não encontrado: {directory}")
        sys.exit(1)
    
    json_files = list(directory_path.glob("sizing_*.json"))
    
    if not json_files:
        print(f"❌ ERRO: Nenhum arquivo sizing_*.json encontrado em {directory}")
        sys.exit(1)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                
            # Aplicar filtros
            if filters.get('models'):
                model_list = [m.strip().lower() for m in filters['models'].split(',')]
                if report['inputs']['model'].lower() not in model_list:
                    continue
            
            if filters.get('server'):
                if report['inputs']['server'].lower() != filters['server'].lower():
                    continue
            
            report['_filename'] = json_file.name
            reports.append(report)
            
        except Exception as e:
            print(f"⚠️  Aviso: Erro ao carregar {json_file.name}: {e}")
            continue
    
    if not reports:
        print(f"❌ ERRO: Nenhum relatório válido encontrado após aplicar filtros")
        sys.exit(1)
    
    return reports


def validate_report_structure(report: Dict[str, Any]) -> bool:
    """Valida presença de campos obrigatórios."""
    required_fields = [
        'inputs', 'scenarios'
    ]
    
    for field in required_fields:
        if field not in report:
            return False
    
    required_inputs = ['model', 'server', 'concurrency', 'effective_context']
    for field in required_inputs:
        if field not in report['inputs']:
            return False
    
    return True


def extract_metrics(report: Dict[str, Any], scenario: str = "recommended") -> Optional[ComparisonMetrics]:
    """Extrai métricas-chave de um relatório para um cenário específico."""
    try:
        if not validate_report_structure(report):
            return None
        
        inputs = report['inputs']
        scenario_data = report['scenarios'].get(scenario, {}).get('results', {})
        
        if not scenario_data:
            return None
        
        # Cálculos derivados
        fixed_model_gib = scenario_data.get('fixed_model_gib', 0)
        vram_total = scenario_data.get('vram_total_node_effective_gib', 0)
        vram_per_session = scenario_data.get('vram_per_session_gib', 0)
        sessions_effective = scenario_data.get('sessions_per_node_effective', 1)
        
        vram_kv_total = vram_per_session * sessions_effective
        vram_overhead = max(0, vram_total - fixed_model_gib - vram_kv_total)
        
        vram_model_pct = (fixed_model_gib / vram_total * 100) if vram_total > 0 else 0
        vram_kv_pct = (vram_kv_total / vram_total * 100) if vram_total > 0 else 0
        
        # Eficiência energética
        total_power = scenario_data.get('total_power_kw_with_storage', scenario_data.get('total_power_kw', 0))
        nodes = scenario_data.get('nodes_final', 1)
        total_sessions = sessions_effective * nodes
        sessions_per_kw = total_sessions / total_power if total_power > 0 else 0
        
        # Custo estimado (premissas do prompt)
        dgx_cost = 500000  # $500k por DGX
        storage = scenario_data.get('storage', {})
        storage_tb = storage.get('storage_total_recommended_tb', 0)
        storage_cost = storage_tb * 200  # $200/TB
        
        capex = (nodes * dgx_cost) + storage_cost
        opex_energy_year = total_power * 8760 * 0.15  # $0.15/kWh, 8760h/ano
        opex_maintenance_year = capex * 0.10  # 10% CapEx/ano
        tco_3years = capex + (opex_energy_year * 3) + (opex_maintenance_year * 3)
        cost_per_session_month = (tco_3years / 36) / total_sessions if total_sessions > 0 else 0
        
        return ComparisonMetrics(
            model=inputs['model'],
            server=inputs['server'],
            concurrency=inputs['concurrency'],
            effective_context=inputs['effective_context'],
            kv_precision=inputs.get('kv_precision', 'N/A'),
            
            vram_per_session_gib=vram_per_session,
            sessions_per_node_capacity=scenario_data.get('sessions_per_node', 0),
            sessions_per_node_effective=sessions_effective,
            
            nodes_final=nodes,
            hbm_utilization_ratio=scenario_data.get('hbm_utilization_ratio_effective', 0),
            
            fixed_model_gib=fixed_model_gib,
            vram_total_node_gib=vram_total,
            vram_model_percent=vram_model_pct,
            vram_kv_percent=vram_kv_pct,
            
            total_power_kw=total_power,
            total_rack_u=scenario_data.get('total_rack_u_with_storage', scenario_data.get('total_rack_u', 0)),
            storage_total_tb=storage_tb,
            
            sessions_per_kw=sessions_per_kw,
            cost_per_session_month=cost_per_session_month
        )
        
    except Exception as e:
        print(f"⚠️  Erro ao extrair métricas: {e}")
        return None


def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formata dados como tabela Markdown."""
    lines = []
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    
    # Rows
    for row in rows:
        formatted_row = []
        for cell in row:
            if isinstance(cell, float):
                formatted_row.append(f"{cell:.2f}")
            else:
                formatted_row.append(str(cell))
        lines.append("| " + " | ".join(formatted_row) + " |")
    
    return "\n".join(lines)


def generate_rankings(metrics_list: List[ComparisonMetrics]) -> Dict[str, List[Tuple[str, float]]]:
    """Gera rankings por métrica."""
    rankings = {}
    
    # KV Efficiency (menor é melhor)
    kv_sorted = sorted(metrics_list, key=lambda m: m.vram_per_session_gib)
    rankings['kv_efficiency'] = [(m.model, m.vram_per_session_gib) for m in kv_sorted]
    
    # Infrastructure Efficiency (menor número de nós é melhor)
    infra_sorted = sorted(metrics_list, key=lambda m: m.nodes_final)
    rankings['infrastructure'] = [(m.model, m.nodes_final) for m in infra_sorted]
    
    # Cost Efficiency (menor custo por sessão)
    cost_sorted = sorted(metrics_list, key=lambda m: m.cost_per_session_month)
    rankings['cost'] = [(m.model, m.cost_per_session_month) for m in cost_sorted]
    
    # Energy Efficiency (maior sessões/kW)
    energy_sorted = sorted(metrics_list, key=lambda m: m.sessions_per_kw, reverse=True)
    rankings['energy'] = [(m.model, m.sessions_per_kw) for m in energy_sorted]
    
    return rankings


def generate_markdown_report(metrics_list: List[ComparisonMetrics], scenario: str, output_path: str):
    """Gera relatório Markdown completo."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append(f"# Análise Comparativa de Sizing de Modelos - {timestamp}")
    lines.append("")
    lines.append("## Resumo Executivo")
    lines.append("")
    
    # Resumo
    model_names = [m.model for m in metrics_list]
    servers = list(set([m.server for m in metrics_list]))
    concurrency = metrics_list[0].concurrency if metrics_list else 0
    context = metrics_list[0].effective_context if metrics_list else 0
    kv_prec = metrics_list[0].kv_precision if metrics_list else "N/A"
    
    lines.append(f"- **Modelos analisados**: {len(metrics_list)} ({', '.join(model_names)})")
    lines.append(f"- **Servidor(es)**: {', '.join(servers)}")
    lines.append(f"- **Concorrência**: {concurrency:,} sessões simultâneas")
    lines.append(f"- **Contexto efetivo**: {context:,} tokens")
    lines.append(f"- **Precisão KV**: {kv_prec}")
    lines.append(f"- **Cenário de referência**: {scenario.upper()}")
    lines.append("")
    
    # Rankings
    rankings = generate_rankings(metrics_list)
    
    lines.append("## 🏆 Rankings de Eficiência")
    lines.append("")
    
    # Ranking KV
    lines.append("### 1. Eficiência de KV Cache (menor é melhor)")
    lines.append("")
    headers = ["Posição", "Modelo", "KV/Sessão (GB)", "Sessões/Nó (Capacidade)", "Observação"]
    rows = []
    medals = ["🥇 1º", "🥈 2º", "🥉 3º"]
    
    for i, m in enumerate(metrics_list[:3]):
        obs = ""
        if i == 0 and len(metrics_list) > 1:
            improvement = ((metrics_list[1].vram_per_session_gib - m.vram_per_session_gib) / 
                          metrics_list[1].vram_per_session_gib * 100)
            obs = f"{improvement:.0f}% mais eficiente"
        
        rows.append([
            medals[i] if i < 3 else f"{i+1}º",
            m.model,
            f"{m.vram_per_session_gib:.3f}",
            m.sessions_per_node_capacity,
            obs
        ])
    
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    
    # Ranking Infraestrutura
    lines.append("### 2. Eficiência de Infraestrutura")
    lines.append("")
    headers = ["Posição", "Modelo", "Nós DGX", "Sessões/Nó", "Utilização HBM", "Storage (TB)"]
    rows = []
    
    infra_sorted = sorted(metrics_list, key=lambda m: m.nodes_final)
    for i, m in enumerate(infra_sorted[:3]):
        rows.append([
            medals[i] if i < 3 else f"{i+1}º",
            m.model,
            m.nodes_final,
            m.sessions_per_node_effective,
            f"{m.hbm_utilization_ratio*100:.1f}%",
            f"{m.storage_total_tb:.2f}"
        ])
    
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    
    # Comparativo de VRAM
    lines.append("### 3. Breakdown de VRAM por Nó")
    lines.append("")
    headers = ["Modelo", "Peso Fixo (GB)", "KV Total (GB)", "VRAM Total (GB)", "% Modelo", "% KV"]
    rows = []
    
    for m in metrics_list:
        kv_total = m.vram_per_session_gib * m.sessions_per_node_effective
        rows.append([
            m.model,
            f"{m.fixed_model_gib:.1f}",
            f"{kv_total:.1f}",
            f"{m.vram_total_node_gib:.1f}",
            f"{m.vram_model_percent:.1f}%",
            f"{m.vram_kv_percent:.1f}%"
        ])
    
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    
    # Recursos Físicos
    lines.append("### 4. Recursos Físicos")
    lines.append("")
    headers = ["Modelo", "Nós", "Energia (kW)", "Rack (U)", "Storage (TB)", "kW/Sessão"]
    rows = []
    
    for m in metrics_list:
        total_sessions = m.sessions_per_node_effective * m.nodes_final
        kw_per_session = m.total_power_kw / total_sessions if total_sessions > 0 else 0
        rows.append([
            m.model,
            m.nodes_final,
            f"{m.total_power_kw:.1f}",
            m.total_rack_u,
            f"{m.storage_total_tb:.2f}",
            f"{kw_per_session:.3f}"
        ])
    
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    
    # TCO
    lines.append("### 5. Análise de Custo (TCO 3 anos)")
    lines.append("")
    lines.append("**Premissas:**")
    lines.append("- Custo por DGX-B300: $500k USD")
    lines.append("- Energia: $0.15/kWh, 24x7")
    lines.append("- Storage NVMe: $200/TB")
    lines.append("- Manutenção: 10% CapEx/ano")
    lines.append("")
    
    headers = ["Modelo", "Nós", "TCO Total (3 anos)", "Custo/Sessão/Mês", "Eficiência Energética"]
    rows = []
    
    cost_sorted = sorted(metrics_list, key=lambda m: m.cost_per_session_month)
    for m in cost_sorted:
        total_sessions = m.sessions_per_node_effective * m.nodes_final
        tco = m.cost_per_session_month * total_sessions * 36
        rows.append([
            m.model,
            m.nodes_final,
            f"${tco/1e6:.2f}M",
            f"${m.cost_per_session_month:.0f}",
            f"{m.sessions_per_kw:.2f} sess/kW"
        ])
    
    lines.append(format_markdown_table(headers, rows))
    lines.append("")
    
    # Recomendação
    lines.append("## 💡 Recomendação Executiva")
    lines.append("")
    
    best_kv = rankings['kv_efficiency'][0][0]
    best_cost = rankings['cost'][0][0]
    best_energy = rankings['energy'][0][0]
    
    lines.append("### Para Produção Crítica (SLA > 99.9%)")
    lines.append(f"**Modelo recomendado**: {best_kv}")
    lines.append(f"**Justificativa**: Melhor eficiência de KV cache, permitindo maior densidade de sessões por nó.")
    lines.append("")
    
    lines.append("### Para Custo Otimizado")
    lines.append(f"**Modelo recomendado**: {best_cost}")
    lines.append(f"**Justificativa**: Menor TCO por sessão simultânea.")
    lines.append("")
    
    lines.append("### Para Eficiência Energética")
    lines.append(f"**Modelo recomendado**: {best_energy}")
    lines.append(f"**Justificativa**: Máximo aproveitamento de energia (sessões por kW).")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Relatório gerado automaticamente pela Calculadora de Sizing de Infraestrutura para Inferência, desenvolvido pelo time de InfraCore de CLOUD.*")
    
    # Salvar
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_json_report(metrics_list: List[ComparisonMetrics], scenario: str, output_path: str):
    """Gera relatório JSON para automação."""
    rankings = generate_rankings(metrics_list)
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "reports_analyzed": len(metrics_list),
            "scenario": scenario
        },
        "rankings": {
            "kv_efficiency": [
                {"rank": i+1, "model": model, "kv_per_session_gib": value}
                for i, (model, value) in enumerate(rankings['kv_efficiency'])
            ],
            "infrastructure": [
                {"rank": i+1, "model": model, "nodes": value}
                for i, (model, value) in enumerate(rankings['infrastructure'])
            ],
            "cost": [
                {"rank": i+1, "model": model, "cost_per_session_month": value}
                for i, (model, value) in enumerate(rankings['cost'])
            ],
            "energy": [
                {"rank": i+1, "model": model, "sessions_per_kw": value}
                for i, (model, value) in enumerate(rankings['energy'])
            ]
        },
        "metrics": [asdict(m) for m in metrics_list]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def print_terminal_summary(metrics_list: List[ComparisonMetrics]):
    """Imprime sumário executivo no terminal."""
    rankings = generate_rankings(metrics_list)
    
    print("=" * 80)
    print("ANÁLISE COMPARATIVA DE SIZING - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print()
    print(f"📊 Relatórios encontrados: {len(metrics_list)}")
    for m in metrics_list:
        print(f"   ✓ {m.model} ({m.server}, {m.concurrency} sessões, {m.kv_precision})")
    print()
    
    print("🏆 TOP 3 RANKINGS")
    print()
    
    print("1️⃣  Eficiência de KV Cache:")
    medals = ["🥇", "🥈", "🥉"]
    for i, (model, value) in enumerate(rankings['kv_efficiency'][:3]):
        print(f"    {medals[i]} {model}: {value:.2f} GB/sessão")
    print()
    
    print("2️⃣  Custo por Sessão (TCO 3 anos):")
    for i, (model, value) in enumerate(rankings['cost'][:3]):
        print(f"    {medals[i]} {model}: ${value:.0f}/sessão/mês")
    print()
    
    print("3️⃣  Eficiência Energética (sessões/kW):")
    for i, (model, value) in enumerate(rankings['energy'][:3]):
        print(f"    {medals[i]} {model}: {value:.2f} sessões/kW")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Análise Comparativa de Sizing de Modelos LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--directory", default="./relatorios", 
                       help="Diretório com arquivos JSON (default: ./relatorios)")
    parser.add_argument("--models", 
                       help="Filtrar modelos específicos (comma-separated)")
    parser.add_argument("--server", 
                       help="Filtrar por servidor específico")
    parser.add_argument("--scenario", default="recommended", 
                       choices=["minimum", "recommended", "ideal"],
                       help="Cenário de referência (default: recommended)")
    parser.add_argument("--output", default="./relatorios", 
                       help="Diretório de saída (default: ./relatorios)")
    parser.add_argument("--format", default="both", 
                       choices=["markdown", "json", "both"],
                       help="Formato de saída (default: both)")
    parser.add_argument("--verbose", action="store_true",
                       help="Modo verboso")
    
    args = parser.parse_args()
    
    # Carregar relatórios
    filters = {
        'models': args.models,
        'server': args.server
    }
    
    if args.verbose:
        print(f"Carregando relatórios de {args.directory}...")
    
    reports = load_sizing_reports(args.directory, filters)
    
    # Extrair métricas
    metrics_list = []
    for report in reports:
        metrics = extract_metrics(report, args.scenario)
        if metrics:
            metrics_list.append(metrics)
        elif args.verbose:
            print(f"⚠️  Não foi possível extrair métricas de {report.get('_filename', 'unknown')}")
    
    if not metrics_list:
        print("❌ ERRO: Nenhuma métrica extraída dos relatórios")
        sys.exit(1)
    
    # Ordenar por KV efficiency
    metrics_list.sort(key=lambda m: m.vram_per_session_gib)
    
    # Sumário no terminal
    print_terminal_summary(metrics_list)
    
    # Gerar relatórios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["markdown", "both"]:
        md_path = output_dir / f"analise_comparativa_{timestamp}.md"
        generate_markdown_report(metrics_list, args.scenario, str(md_path))
        print(f"✅ Relatório Markdown gerado: {md_path}")
    
    if args.format in ["json", "both"]:
        json_path = output_dir / f"analise_comparativa_{timestamp}.json"
        generate_json_report(metrics_list, args.scenario, str(json_path))
        print(f"✅ Relatório JSON gerado: {json_path}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
