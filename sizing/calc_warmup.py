"""
Cálculo de workload operacional: warmup/cold start de modelos.

Estima tempo necessário para carregar artefatos do modelo do storage para memória,
considerando throughput e IOPS disponíveis, pattern de acesso (seq/rand) e concorrência.
"""

from dataclasses import dataclass
from typing import Dict, Any
from .storage import StorageProfile


@dataclass
class WarmupEstimate:
    """Estimativa de tempo de warmup/cold start."""
    
    # Inputs
    artifact_size_gib: float
    artifact_size_mib: float
    warmup_concurrency: int
    read_pattern: str  # "seq" ou "rand"
    utilization_ratio: float
    
    # Storage efetivo
    throughput_effective_mbps: float
    iops_effective: int
    block_size_kb: float
    
    # Estimativas
    warmup_time_per_pod_s: float
    warmup_time_cluster_s: float  # Com concorrência
    warmup_time_by_iops_s: float  # Limitado por IOPS (random)
    warmup_time_final_s: float    # Tempo real estimado
    
    # Análise
    bottleneck: str  # "throughput-limited" ou "iops-limited"
    rationale: Dict[str, Any]


def calc_warmup_estimate(
    storage: StorageProfile,
    artifact_size_gib: float,
    warmup_concurrency: int = 1,
    read_pattern: str = "seq",
    utilization_ratio: float = 0.8
) -> WarmupEstimate:
    """
    Calcula estimativa de tempo de warmup/cold start.
    
    Args:
        storage: Perfil de storage
        artifact_size_gib: Tamanho do artefato do modelo em GiB
        warmup_concurrency: Número de pods iniciando em paralelo
        read_pattern: "seq" (sequencial) ou "rand" (random)
        utilization_ratio: Fração do max utilizável (ex.: 0.8 = 80%)
    
    Returns:
        WarmupEstimate com tempos e análise
    """
    # Converter tamanho para MiB (base 1024)
    artifact_size_mib = artifact_size_gib * 1024.0
    
    # Calcular throughput efetivo
    # Usar throughput teórico (IOPS × BlockSize) se menor que informado
    throughput_theoretical_mbps = (storage.iops_read_max * storage.block_size_kb_read) / 1024.0
    throughput_base_mbps = min(storage.throughput_read_mbps, throughput_theoretical_mbps)
    throughput_effective_mbps = throughput_base_mbps * utilization_ratio
    
    # Calcular IOPS efetivo
    iops_effective = int(storage.iops_read_max * utilization_ratio)
    
    # 1) Tempo por pod (individual, sem concorrência)
    if throughput_effective_mbps > 0:
        warmup_time_per_pod_s = artifact_size_mib / throughput_effective_mbps
    else:
        warmup_time_per_pod_s = 0.0
    
    # 2) Tempo com concorrência (assumindo storage como gargalo compartilhado)
    if throughput_effective_mbps > 0:
        total_data_mib = artifact_size_mib * warmup_concurrency
        warmup_time_cluster_s = total_data_mib / throughput_effective_mbps
    else:
        warmup_time_cluster_s = 0.0
    
    # 3) Se pattern é random, validar por IOPS
    warmup_time_by_iops_s = 0.0
    if read_pattern == "rand" and storage.block_size_kb_read > 0:
        # Calcular número de IOs necessários
        bytes_per_io_mib = storage.block_size_kb_read / 1024.0
        total_ios = artifact_size_mib / bytes_per_io_mib
        
        if iops_effective > 0:
            # Tempo para completar todas as IOs (1 pod)
            warmup_time_by_iops_s = total_ios / iops_effective
            
            # Com concorrência
            total_ios_cluster = total_ios * warmup_concurrency
            warmup_time_by_iops_cluster_s = total_ios_cluster / iops_effective
            
            # Para random, usar o máximo entre throughput e IOPS
            warmup_time_cluster_s = max(warmup_time_cluster_s, warmup_time_by_iops_cluster_s)
            warmup_time_by_iops_s = warmup_time_by_iops_cluster_s
    
    # 4) Determinar tempo final e bottleneck
    if read_pattern == "rand" and warmup_time_by_iops_s > warmup_time_cluster_s * 1.1:
        # IOPS é gargalo (>10% mais lento)
        warmup_time_final_s = warmup_time_by_iops_s
        bottleneck = "iops-limited"
    else:
        # Throughput é gargalo
        warmup_time_final_s = warmup_time_cluster_s
        bottleneck = "throughput-limited"
    
    # Racional
    rationale = {
        "formula_throughput": "warmup_time_s = (artifact_size_mib × concurrency) / throughput_effective_mbps",
        "formula_iops": "warmup_time_s = (total_ios × concurrency) / iops_effective (se pattern==rand)",
        "inputs": {
            "artifact_size_gib": round(artifact_size_gib, 2),
            "artifact_size_mib": round(artifact_size_mib, 2),
            "warmup_concurrency": warmup_concurrency,
            "read_pattern": read_pattern,
            "utilization_ratio": utilization_ratio,
            "storage_throughput_read_mbps": storage.throughput_read_mbps,
            "storage_iops_read_max": storage.iops_read_max,
            "storage_block_size_kb_read": storage.block_size_kb_read,
            "throughput_effective_mbps": round(throughput_effective_mbps, 2),
            "iops_effective": iops_effective
        },
        "calculation": {
            "warmup_time_per_pod_s": round(warmup_time_per_pod_s, 2),
            "warmup_time_cluster_throughput_s": round(warmup_time_cluster_s, 2),
            "warmup_time_cluster_iops_s": round(warmup_time_by_iops_s, 2) if read_pattern == "rand" else "N/A",
            "bottleneck": bottleneck
        },
        "assumption": (
            f"Storage é gargalo compartilhado. {warmup_concurrency} pods leem simultaneamente. "
            f"Utilização real: {utilization_ratio*100:.0f}% do máximo teórico. "
            f"{'IOPS limita acesso random.' if read_pattern == 'rand' else 'Throughput limita acesso sequencial.'}"
        ),
        "operational_meaning": (
            f"Tempo estimado para carregar modelo ({artifact_size_gib:.1f} GiB) do storage: "
            f"{warmup_time_final_s:.1f}s por pod, {warmup_time_final_s:.1f}s para {warmup_concurrency} pods simultâneos. "
            f"Bottleneck: {bottleneck}. "
            "Impacta tempo de recuperação após falha e scale-out velocity."
        )
    }
    
    return WarmupEstimate(
        artifact_size_gib=artifact_size_gib,
        artifact_size_mib=artifact_size_mib,
        warmup_concurrency=warmup_concurrency,
        read_pattern=read_pattern,
        utilization_ratio=utilization_ratio,
        throughput_effective_mbps=throughput_effective_mbps,
        iops_effective=iops_effective,
        block_size_kb=storage.block_size_kb_read,
        warmup_time_per_pod_s=warmup_time_per_pod_s,
        warmup_time_cluster_s=warmup_time_cluster_s,
        warmup_time_by_iops_s=warmup_time_by_iops_s,
        warmup_time_final_s=warmup_time_final_s,
        bottleneck=bottleneck,
        rationale=rationale
    )


def format_warmup_report(warmup: WarmupEstimate) -> str:
    """
    Formata relatório de warmup em texto.
    
    Args:
        warmup: Estimativa de warmup
    
    Returns:
        String formatada
    """
    lines = []
    
    lines.append("WARMUP / COLD START ESTIMATE:")
    lines.append("=" * 100)
    lines.append(f"Artifact Size:        {warmup.artifact_size_gib:.2f} GiB ({warmup.artifact_size_mib:.1f} MiB)")
    lines.append(f"Concurrency:          {warmup.warmup_concurrency} pods")
    lines.append(f"Read Pattern:         {warmup.read_pattern.upper()}")
    lines.append(f"Utilization Ratio:    {warmup.utilization_ratio*100:.0f}%")
    lines.append("")
    
    lines.append("STORAGE EFFECTIVE:")
    lines.append(f"  • Throughput:       {warmup.throughput_effective_mbps:.1f} MB/s")
    lines.append(f"  • IOPS:             {warmup.iops_effective:,}")
    lines.append(f"  • Block Size:       {warmup.block_size_kb:.1f} KB")
    lines.append("")
    
    lines.append("WARMUP TIME ESTIMATES:")
    lines.append(f"  • Per pod:          {warmup.warmup_time_per_pod_s:.1f} seconds ({warmup.warmup_time_per_pod_s/60:.2f} min)")
    lines.append(f"  • Cluster ({warmup.warmup_concurrency} pods): {warmup.warmup_time_final_s:.1f} seconds ({warmup.warmup_time_final_s/60:.2f} min)")
    lines.append(f"  • Bottleneck:       {warmup.bottleneck.upper()}")
    lines.append("")
    
    lines.append("OPERATIONAL IMPACT:")
    lines.append(f"  {warmup.rationale['operational_meaning']}")
    lines.append("")
    
    return "\n".join(lines)


def warmup_to_dict(warmup: WarmupEstimate) -> Dict[str, Any]:
    """
    Converte warmup para dicionário (para JSON).
    
    Args:
        warmup: Estimativa de warmup
    
    Returns:
        Dict serializável
    """
    return {
        "inputs": {
            "artifact_size_gib": round(warmup.artifact_size_gib, 2),
            "warmup_concurrency": warmup.warmup_concurrency,
            "read_pattern": warmup.read_pattern,
            "utilization_ratio": warmup.utilization_ratio
        },
        "storage_effective": {
            "throughput_mbps": round(warmup.throughput_effective_mbps, 2),
            "iops": warmup.iops_effective,
            "block_size_kb": round(warmup.block_size_kb, 2)
        },
        "estimates": {
            "warmup_time_per_pod_s": round(warmup.warmup_time_per_pod_s, 2),
            "warmup_time_cluster_s": round(warmup.warmup_time_final_s, 2),
            "warmup_time_per_pod_min": round(warmup.warmup_time_per_pod_s / 60, 2),
            "warmup_time_cluster_min": round(warmup.warmup_time_final_s / 60, 2),
            "bottleneck": warmup.bottleneck
        },
        "rationale": warmup.rationale
    }
