"""
Módulo de Profile de Storage da Plataforma

Define e valida os requisitos de storage estrutural da plataforma de IA
(Sistema Operacional, NVIDIA AI Enterprise, Runtime, etc.) por servidor.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class PlatformStorageProfile:
    """
    Profile de storage estrutural da plataforma por servidor.
    
    Attributes:
        os_installation_gb: Volume para Sistema Operacional (Ubuntu/RHEL + drivers)
        nvidia_ai_enterprise_gb: Volume para NVIDIA AI Enterprise (CUDA, cuDNN, TensorRT, Triton)
        container_runtime_gb: Volume para runtime de containers (Docker/containerd, K8s, imagens)
        model_runtime_engines_gb: Volume para engines de inferência (TensorRT-LLM, vLLM, NIM)
        platform_dependencies_gb: Volume para dependências (libs Python, NCCL, ML libs)
        config_and_metadata_gb: Volume para config (Helm charts, manifests, certs, secrets)
        notes: Notas explicativas sobre os volumes
        source: Origem do profile (arquivo)
    """
    
    os_installation_gb: float
    nvidia_ai_enterprise_gb: float
    container_runtime_gb: float
    model_runtime_engines_gb: float
    platform_dependencies_gb: float
    config_and_metadata_gb: float
    notes: str = ""
    source: str = "platform_storage_profile.json"
    
    def validate(self) -> None:
        """Valida o profile de storage da plataforma."""
        fields = [
            ("os_installation_gb", self.os_installation_gb),
            ("nvidia_ai_enterprise_gb", self.nvidia_ai_enterprise_gb),
            ("container_runtime_gb", self.container_runtime_gb),
            ("model_runtime_engines_gb", self.model_runtime_engines_gb),
            ("platform_dependencies_gb", self.platform_dependencies_gb),
            ("config_and_metadata_gb", self.config_and_metadata_gb)
        ]
        
        for field_name, value in fields:
            if value < 0:
                raise ValueError(
                    f"❌ ERRO: Campo '{field_name}' não pode ser negativo: {value} GB"
                )
    
    @property
    def total_per_server_gb(self) -> float:
        """Volume total da plataforma por servidor em GB."""
        return (
            self.os_installation_gb +
            self.nvidia_ai_enterprise_gb +
            self.container_runtime_gb +
            self.model_runtime_engines_gb +
            self.platform_dependencies_gb +
            self.config_and_metadata_gb
        )
    
    @property
    def total_per_server_tb(self) -> float:
        """Volume total da plataforma por servidor em TB."""
        return self.total_per_server_gb / 1024.0
    
    def calc_total_platform_volume_tb(self, num_nodes: int) -> float:
        """
        Calcula o volume total da plataforma para N nós.
        
        Args:
            num_nodes: Número de servidores DGX
        
        Returns:
            Volume total em TB
        """
        if num_nodes <= 0:
            raise ValueError(
                f"❌ ERRO: Número de nós deve ser > 0: {num_nodes}"
            )
        
        return self.total_per_server_tb * num_nodes
    
    def get_breakdown(self) -> Dict[str, float]:
        """
        Retorna breakdown dos volumes por componente.
        
        Returns:
            Dict com componente -> volume em GB
        """
        return {
            "Sistema Operacional": self.os_installation_gb,
            "NVIDIA AI Enterprise": self.nvidia_ai_enterprise_gb,
            "Runtime de Containers": self.container_runtime_gb,
            "Engines de Inferência": self.model_runtime_engines_gb,
            "Dependências da Plataforma": self.platform_dependencies_gb,
            "Configuração e Metadados": self.config_and_metadata_gb,
            "TOTAL por servidor": self.total_per_server_gb
        }
    
    def get_rationale(self, num_nodes: int) -> Dict[str, Any]:
        """
        Retorna racional de cálculo do volume da plataforma.
        
        Args:
            num_nodes: Número de nós
        
        Returns:
            Dict com fórmula, inputs, assumption e operational_meaning
        """
        total_tb = self.calc_total_platform_volume_tb(num_nodes)
        
        return {
            "formula": "platform_volume_total_tb = total_per_server_tb × num_nodes",
            "inputs": {
                "os_installation_gb": self.os_installation_gb,
                "nvidia_ai_enterprise_gb": self.nvidia_ai_enterprise_gb,
                "container_runtime_gb": self.container_runtime_gb,
                "model_runtime_engines_gb": self.model_runtime_engines_gb,
                "platform_dependencies_gb": self.platform_dependencies_gb,
                "config_and_metadata_gb": self.config_and_metadata_gb,
                "total_per_server_gb": round(self.total_per_server_gb, 2),
                "total_per_server_tb": round(self.total_per_server_tb, 3),
                "num_nodes": num_nodes,
                "total_tb": round(total_tb, 3)
            },
            "assumption": f"Volume estrutural fixo por servidor DGX. Não inclui pesos de modelos ou dados dinâmicos. "
                         f"Considera instalação completa do SO, NVIDIA AI Enterprise, runtime e engines de inferência.",
            "operational_meaning": f"Cada servidor DGX requer {self.total_per_server_tb:.2f} TB para a pilha completa de software. "
                                  f"Total de {total_tb:.2f} TB para {num_nodes} nó(s). "
                                  f"Este volume é fixo e não varia com concorrência ou modelo. "
                                  f"Crítico para boot, runtime e operação da plataforma."
        }


def load_platform_storage_profile(
    filepath: str = "platform_storage_profile.json"
) -> PlatformStorageProfile:
    """
    Carrega profile de storage da plataforma do arquivo JSON.
    
    Args:
        filepath: Caminho para o arquivo de profile
    
    Returns:
        PlatformStorageProfile validado
    
    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o profile for inválido
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ ERRO: Arquivo de profile de storage da plataforma não encontrado: {filepath}\n\n"
            f"Este arquivo é obrigatório e define os volumes estruturais por servidor.\n"
            f"Crie o arquivo com o seguinte conteúdo mínimo:\n\n"
            f'{{\n'
            f'  "os_installation_gb": 120,\n'
            f'  "nvidia_ai_enterprise_gb": 250,\n'
            f'  "container_runtime_gb": 80,\n'
            f'  "model_runtime_engines_gb": 150,\n'
            f'  "platform_dependencies_gb": 50,\n'
            f'  "config_and_metadata_gb": 25,\n'
            f'  "notes": "Volumes para plataforma NVIDIA AI Enterprise completa."\n'
            f'}}\n'
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ ERRO: Arquivo {filepath} não é um JSON válido: {e}")
    
    # Validar campos obrigatórios
    required_fields = [
        "os_installation_gb",
        "nvidia_ai_enterprise_gb",
        "container_runtime_gb",
        "model_runtime_engines_gb",
        "platform_dependencies_gb",
        "config_and_metadata_gb"
    ]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"❌ ERRO: Campo obrigatório '{field}' ausente em {filepath}"
            )
    
    profile = PlatformStorageProfile(
        os_installation_gb=data["os_installation_gb"],
        nvidia_ai_enterprise_gb=data["nvidia_ai_enterprise_gb"],
        container_runtime_gb=data["container_runtime_gb"],
        model_runtime_engines_gb=data["model_runtime_engines_gb"],
        platform_dependencies_gb=data["platform_dependencies_gb"],
        config_and_metadata_gb=data["config_and_metadata_gb"],
        notes=data.get("notes", ""),
        source=filepath
    )
    
    # Validar
    profile.validate()
    
    return profile
