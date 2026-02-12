"""
Módulo de Política de Capacidade

Define e valida políticas de margem de capacidade para storage.
Essa margem protege contra subdimensionamento e permite:
- Crescimento orgânico da plataforma
- Retenção adicional de logs não prevista
- Expansão futura sem reengenharia
- Resiliência operacional
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class CapacityPolicy:
    """
    Política de margem de capacidade para storage.
    
    Attributes:
        margin_percent: Percentual de margem (0.0 a 1.0). Ex: 0.50 = 50%
        apply_to: Lista de métricas às quais aplicar a margem
        target_load_time_sec: Tempo máximo (segundos) para carregar modelo no restart
        notes: Justificativa da política
        source: Origem da política (arquivo ou CLI override)
    """
    
    margin_percent: float
    apply_to: List[str]
    target_load_time_sec: float = 60.0  # Default: 60 segundos
    notes: str = ""
    source: str = "parameters.json"
    
    def validate(self) -> None:
        """Valida a política de capacidade."""
        if self.margin_percent < 0:
            raise ValueError(
                f"capacity_margin_percent não pode ser negativo: {self.margin_percent}"
            )
        
        if self.margin_percent > 1.0:
            raise ValueError(
                f"capacity_margin_percent não pode ser maior que 1.0 (100%): {self.margin_percent}. "
                f"Use valores entre 0.0 e 1.0 (ex: 0.50 para 50%)"
            )
        
        # Validação de target_load_time_sec
        if self.target_load_time_sec <= 0:
            raise ValueError(
                f"target_load_time_sec deve ser > 0: {self.target_load_time_sec} segundos"
            )
        
        if self.target_load_time_sec < 10:
            raise ValueError(
                f"target_load_time_sec muito baixo: {self.target_load_time_sec}s. "
                f"Use valores >= 10 segundos para garantir viabilidade com storage real."
            )
        
        valid_targets = {
            "storage_total",
            "storage_model",
            "storage_cache",
            "storage_logs",
            "storage_operational"
        }
        
        for target in self.apply_to:
            if target not in valid_targets:
                raise ValueError(
                    f"Métrica inválida em 'apply_margin_to': {target}. "
                    f"Valores válidos: {sorted(valid_targets)}"
                )
    
    def apply_margin(self, base_value: float, metric_name: str) -> float:
        """
        Aplica margem a um valor base se a métrica estiver na lista.
        
        Args:
            base_value: Valor base calculado
            metric_name: Nome da métrica (ex: "storage_total")
        
        Returns:
            Valor recomendado com margem aplicada (ou valor base se não aplicável)
        """
        if metric_name in self.apply_to:
            recommended = base_value * (1 + self.margin_percent)
            # Arredondar para 2 casas decimais
            return round(recommended, 2)
        else:
            return base_value
    
    def get_margin_info(self, base_value: float, metric_name: str) -> Dict[str, Any]:
        """
        Retorna informações completas sobre a margem aplicada.
        
        Returns:
            Dict com base_value, recommended_value, margin_applied, margin_percent
        """
        recommended = self.apply_margin(base_value, metric_name)
        margin_applied = metric_name in self.apply_to
        
        return {
            "base_value": round(base_value, 2),
            "recommended_value": round(recommended, 2),
            "margin_applied": margin_applied,
            "margin_percent": self.margin_percent if margin_applied else 0.0,
            "metric_name": metric_name
        }


def load_capacity_policy(
    filepath: str = "parameters.json",
    override_margin: float = None
) -> CapacityPolicy:
    """
    Carrega política de capacidade do arquivo JSON.
    
    Args:
        filepath: Caminho para o arquivo de política
        override_margin: Percentual de override via CLI (opcional)
    
    Returns:
        CapacityPolicy validada
    
    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se a política for inválida
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ ERRO: Arquivo de política de capacidade não encontrado: {filepath}\n\n"
            f"Este arquivo é obrigatório e define a margem estratégica de capacidade.\n"
            f"Crie o arquivo com o seguinte conteúdo mínimo:\n\n"
            f'{{\n'
            f'  "capacity_margin_percent": 0.50,\n'
            f'  "apply_margin_to": ["storage_total", "storage_model", "storage_cache", "storage_logs"],\n'
            f'  "target_load_time_sec": 60.0,\n'
            f'  "notes": "Margem recomendada para crescimento e eventos não previstos."\n'
            f'}}\n'
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ ERRO: Arquivo {filepath} não é um JSON válido: {e}")
    
    # Validar campos obrigatórios
    if "capacity_margin_percent" not in data:
        raise ValueError(
            f"❌ ERRO: Campo obrigatório 'capacity_margin_percent' ausente em {filepath}"
        )
    
    if "apply_margin_to" not in data:
        raise ValueError(
            f"❌ ERRO: Campo obrigatório 'apply_margin_to' ausente em {filepath}"
        )
    
    # Aplicar override se fornecido
    margin_percent = override_margin if override_margin is not None else data["capacity_margin_percent"]
    source = f"CLI override (--capacity-margin {override_margin})" if override_margin is not None else filepath
    
    # Carregar target_load_time_sec (com default se não existir para retrocompatibilidade)
    target_load_time_sec = data.get("target_load_time_sec", 60.0)
    
    policy = CapacityPolicy(
        margin_percent=margin_percent,
        apply_to=data["apply_margin_to"],
        target_load_time_sec=target_load_time_sec,
        notes=data.get("notes", ""),
        source=source
    )
    
    # Validar
    policy.validate()
    
    return policy
