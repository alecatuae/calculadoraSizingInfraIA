"""
Validação de consistência física entre IOPS, Throughput e Block Size.

Formula base (obrigatória):
Throughput(MB/s) = (IOPS × BlockSize(KB)) / 1024

Esta validação garante que os parâmetros de storage no JSON são fisicamente consistentes.
"""

from dataclasses import dataclass
from typing import Dict, Any
from .storage import StorageProfile


# Thresholds de validação
WARNING_THRESHOLD = 0.10  # 10%
ERROR_THRESHOLD = 0.25    # 25%


@dataclass
class StorageValidationResult:
    """Resultado da validação de um eixo (read ou write)."""
    
    # Valores informados
    iops_informed: int
    throughput_mbps_informed: float
    block_size_kb_informed: float
    
    # Valores calculados
    throughput_mbps_calculated: float  # A partir de IOPS e Block Size
    iops_calculated: int               # A partir de Throughput e Block Size
    block_size_kb_calculated: float    # A partir de Throughput e IOPS
    
    # Divergências
    throughput_divergence_pct: float
    iops_divergence_pct: float
    block_size_divergence_pct: float
    
    # Status
    status: str  # "ok", "warning", "error"
    messages: list[str]


@dataclass
class StorageProfileValidation:
    """Validação completa de um perfil de storage."""
    
    profile_name: str
    read_validation: StorageValidationResult
    write_validation: StorageValidationResult
    overall_status: str  # "ok", "warning", "error"
    messages: list[str]


def validate_storage_profile(profile: StorageProfile) -> StorageProfileValidation:
    """
    Valida consistência física de um perfil de storage.
    
    Args:
        profile: Perfil de storage a validar
    
    Returns:
        StorageProfileValidation com resultados e status
    """
    # Validar leitura
    read_val = _validate_axis(
        iops=profile.iops_read_max,
        throughput_mbps=profile.throughput_read_mbps,
        block_size_kb=profile.block_size_kb_read,
        axis_name="read"
    )
    
    # Validar escrita
    write_val = _validate_axis(
        iops=profile.iops_write_max,
        throughput_mbps=profile.throughput_write_mbps,
        block_size_kb=profile.block_size_kb_write,
        axis_name="write"
    )
    
    # Status geral
    statuses = [read_val.status, write_val.status]
    if "error" in statuses:
        overall_status = "error"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "ok"
    
    # Mensagens gerais
    messages = []
    if overall_status == "error":
        messages.append(
            f"❌ Divergência crítica (>{ERROR_THRESHOLD*100:.0f}%) entre IOPS/Throughput/BlockSize "
            f"no perfil '{profile.name}'."
        )
    elif overall_status == "warning":
        messages.append(
            f"⚠️ Divergência moderada ({WARNING_THRESHOLD*100:.0f}%-{ERROR_THRESHOLD*100:.0f}%) "
            f"no perfil '{profile.name}'. Recomenda-se revisar valores."
        )
    
    return StorageProfileValidation(
        profile_name=profile.name,
        read_validation=read_val,
        write_validation=write_val,
        overall_status=overall_status,
        messages=messages
    )


def _validate_axis(
    iops: int,
    throughput_mbps: float,
    block_size_kb: float,
    axis_name: str
) -> StorageValidationResult:
    """
    Valida consistência de um eixo (read ou write).
    
    Formula: Throughput(MB/s) = (IOPS × BlockSize(KB)) / 1024
    
    Args:
        iops: IOPS máximos informados
        throughput_mbps: Throughput em MB/s informado
        block_size_kb: Block size em KB informado
        axis_name: "read" ou "write"
    
    Returns:
        StorageValidationResult
    """
    # Calcular valores teóricos
    # 1) Throughput a partir de IOPS e Block Size
    throughput_calculated = (iops * block_size_kb) / 1024.0
    
    # 2) IOPS a partir de Throughput e Block Size
    if block_size_kb > 0:
        iops_calculated = int((throughput_mbps * 1024.0) / block_size_kb)
    else:
        iops_calculated = 0
    
    # 3) Block Size a partir de Throughput e IOPS
    if iops > 0:
        block_size_calculated = (throughput_mbps * 1024.0) / iops
    else:
        block_size_calculated = 0.0
    
    # Calcular divergências
    throughput_div = _calc_divergence(throughput_mbps, throughput_calculated)
    iops_div = _calc_divergence(iops, iops_calculated)
    block_size_div = _calc_divergence(block_size_kb, block_size_calculated)
    
    # Determinar status (usar a maior divergência)
    max_div = max(throughput_div, iops_div, block_size_div)
    
    if max_div > ERROR_THRESHOLD:
        status = "error"
    elif max_div > WARNING_THRESHOLD:
        status = "warning"
    else:
        status = "ok"
    
    # Gerar mensagens
    messages = []
    
    if status == "error":
        messages.append(
            f"❌ [{axis_name.upper()}] Divergência crítica (>{ERROR_THRESHOLD*100:.0f}%)"
        )
        if throughput_div == max_div:
            messages.append(
                f"   Throughput informado: {throughput_mbps:.1f} MB/s | "
                f"Calculado (IOPS×BS/1024): {throughput_calculated:.1f} MB/s | "
                f"Divergência: {throughput_div*100:.1f}%"
            )
        if iops_div == max_div:
            messages.append(
                f"   IOPS informado: {iops:,} | "
                f"Calculado (TP×1024/BS): {iops_calculated:,} | "
                f"Divergência: {iops_div*100:.1f}%"
            )
        if block_size_div == max_div:
            messages.append(
                f"   Block Size informado: {block_size_kb:.1f} KB | "
                f"Calculado (TP×1024/IOPS): {block_size_calculated:.1f} KB | "
                f"Divergência: {block_size_div*100:.1f}%"
            )
    
    elif status == "warning":
        messages.append(
            f"⚠️ [{axis_name.upper()}] Divergência moderada ({WARNING_THRESHOLD*100:.0f}%-{ERROR_THRESHOLD*100:.0f}%)"
        )
        if throughput_div > WARNING_THRESHOLD:
            messages.append(
                f"   Throughput: {throughput_mbps:.1f} MB/s (informado) vs "
                f"{throughput_calculated:.1f} MB/s (calculado) → {throughput_div*100:.1f}%"
            )
        if iops_div > WARNING_THRESHOLD:
            messages.append(
                f"   IOPS: {iops:,} (informado) vs "
                f"{iops_calculated:,} (calculado) → {iops_div*100:.1f}%"
            )
        if block_size_div > WARNING_THRESHOLD:
            messages.append(
                f"   Block Size: {block_size_kb:.1f} KB (informado) vs "
                f"{block_size_calculated:.1f} KB (calculado) → {block_size_div*100:.1f}%"
            )
    
    return StorageValidationResult(
        iops_informed=iops,
        throughput_mbps_informed=throughput_mbps,
        block_size_kb_informed=block_size_kb,
        throughput_mbps_calculated=throughput_calculated,
        iops_calculated=iops_calculated,
        block_size_kb_calculated=block_size_calculated,
        throughput_divergence_pct=throughput_div,
        iops_divergence_pct=iops_div,
        block_size_divergence_pct=block_size_div,
        status=status,
        messages=messages
    )


def _calc_divergence(informed: float, calculated: float) -> float:
    """
    Calcula divergência percentual entre valor informado e calculado.
    
    diff_pct = abs(informado - calculado) / max(informado, calculado)
    
    Args:
        informed: Valor informado
        calculated: Valor calculado
    
    Returns:
        Divergência como fração (0.0 - 1.0)
    """
    if informed == 0 and calculated == 0:
        return 0.0
    
    max_val = max(abs(informed), abs(calculated))
    if max_val == 0:
        return 0.0
    
    return abs(informed - calculated) / max_val


def format_validation_report(validation: StorageProfileValidation) -> str:
    """
    Formata relatório de validação em texto.
    
    Args:
        validation: Resultado da validação
    
    Returns:
        String formatada para inclusão em relatório
    """
    lines = []
    
    lines.append(f"Storage Profile: {validation.profile_name}")
    lines.append(f"Status Geral: {validation.overall_status.upper()}")
    lines.append("")
    
    # Tabela READ
    lines.append("READ VALIDATION:")
    lines.append("=" * 100)
    lines.append(f"{'Métrica':<25} {'Informado':>20} {'Calculado':>20} {'Divergência':>15} {'Status':>15}")
    lines.append("-" * 100)
    
    rv = validation.read_validation
    lines.append(
        f"{'IOPS':<25} {rv.iops_informed:>20,} {rv.iops_calculated:>20,} "
        f"{rv.iops_divergence_pct*100:>14.1f}% {rv.status:>15}"
    )
    lines.append(
        f"{'Block Size (KB)':<25} {rv.block_size_kb_informed:>20,.1f} {rv.block_size_kb_calculated:>20,.1f} "
        f"{rv.block_size_divergence_pct*100:>14.1f}% {rv.status:>15}"
    )
    lines.append(
        f"{'Throughput (MB/s)':<25} {rv.throughput_mbps_informed:>20,.1f} {rv.throughput_mbps_calculated:>20,.1f} "
        f"{rv.throughput_divergence_pct*100:>14.1f}% {rv.status:>15}"
    )
    lines.append("")
    
    # Tabela WRITE
    lines.append("WRITE VALIDATION:")
    lines.append("=" * 100)
    lines.append(f"{'Métrica':<25} {'Informado':>20} {'Calculado':>20} {'Divergência':>15} {'Status':>15}")
    lines.append("-" * 100)
    
    wv = validation.write_validation
    lines.append(
        f"{'IOPS':<25} {wv.iops_informed:>20,} {wv.iops_calculated:>20,} "
        f"{wv.iops_divergence_pct*100:>14.1f}% {wv.status:>15}"
    )
    lines.append(
        f"{'Block Size (KB)':<25} {wv.block_size_kb_informed:>20,.1f} {wv.block_size_kb_calculated:>20,.1f} "
        f"{wv.block_size_divergence_pct*100:>14.1f}% {wv.status:>15}"
    )
    lines.append(
        f"{'Throughput (MB/s)':<25} {wv.throughput_mbps_informed:>20,.1f} {wv.throughput_mbps_calculated:>20,.1f} "
        f"{wv.throughput_divergence_pct*100:>14.1f}% {wv.status:>15}"
    )
    lines.append("")
    
    # Mensagens
    if validation.messages:
        lines.append("MENSAGENS:")
        for msg in validation.messages:
            lines.append(f"  {msg}")
        lines.append("")
    
    if rv.messages:
        for msg in rv.messages:
            lines.append(f"  {msg}")
    
    if wv.messages:
        for msg in wv.messages:
            lines.append(f"  {msg}")
    
    return "\n".join(lines)


def validation_to_dict(validation: StorageProfileValidation) -> Dict[str, Any]:
    """
    Converte validação para dicionário (para JSON).
    
    Args:
        validation: Resultado da validação
    
    Returns:
        Dict serializável
    """
    return {
        "profile_name": validation.profile_name,
        "overall_status": validation.overall_status,
        "read": {
            "iops_informed": validation.read_validation.iops_informed,
            "iops_calculated": validation.read_validation.iops_calculated,
            "iops_divergence_pct": round(validation.read_validation.iops_divergence_pct, 4),
            "throughput_mbps_informed": round(validation.read_validation.throughput_mbps_informed, 2),
            "throughput_mbps_calculated": round(validation.read_validation.throughput_mbps_calculated, 2),
            "throughput_divergence_pct": round(validation.read_validation.throughput_divergence_pct, 4),
            "block_size_kb_informed": round(validation.read_validation.block_size_kb_informed, 2),
            "block_size_kb_calculated": round(validation.read_validation.block_size_kb_calculated, 2),
            "block_size_divergence_pct": round(validation.read_validation.block_size_divergence_pct, 4),
            "status": validation.read_validation.status,
            "messages": validation.read_validation.messages
        },
        "write": {
            "iops_informed": validation.write_validation.iops_informed,
            "iops_calculated": validation.write_validation.iops_calculated,
            "iops_divergence_pct": round(validation.write_validation.iops_divergence_pct, 4),
            "throughput_mbps_informed": round(validation.write_validation.throughput_mbps_informed, 2),
            "throughput_mbps_calculated": round(validation.write_validation.throughput_mbps_calculated, 2),
            "throughput_divergence_pct": round(validation.write_validation.throughput_divergence_pct, 4),
            "block_size_kb_informed": round(validation.write_validation.block_size_kb_informed, 2),
            "block_size_kb_calculated": round(validation.write_validation.block_size_kb_calculated, 2),
            "block_size_divergence_pct": round(validation.write_validation.block_size_divergence_pct, 4),
            "status": validation.write_validation.status,
            "messages": validation.write_validation.messages
        },
        "messages": validation.messages
    }
