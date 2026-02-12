"""
Carregador de configurações JSON (models, servers, storage) com validação de schema.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

from .models import ModelSpec
from .servers import (
    ServerSpec, GPUSpec, PowerSpec, ThermalSpec, CPUSpec, SystemMemorySpec,
    CoolingSpec, StorageSpec, NetworkingSpec, SoftwareSpec, PhysicalSpec,
    PowerSupplySpec, MaxCurrentSpec, DimensionsSpec
)
from .storage import StorageProfile
from .validator import validate_models, validate_servers, validate_storage_profiles


class ConfigLoader:
    """Carrega e gerencia especificações de models, servers e storage com validação."""
    
    def __init__(self, base_path: str = ".", validate: bool = True):
        """
        Args:
            base_path: Caminho base para os arquivos JSON
            validate: Se True, valida schemas ao carregar
        """
        self.base_path = Path(base_path)
        self.validate = validate
        
        # Cache
        self._models: Dict[str, ModelSpec] = {}
        self._servers: Dict[str, ServerSpec] = {}
        self._storage: Dict[str, StorageProfile] = {}
        
        # Dados brutos para validação
        self._models_data: List[Dict[str, Any]] = []
        self._servers_data: List[Dict[str, Any]] = []
        self._storage_data: List[Dict[str, Any]] = []
    
    def load_models(self, filepath: str = "models.json") -> Dict[str, ModelSpec]:
        """Carrega especificações de modelos do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Arquivo de modelos não encontrado: {path}\n"
                "Certifique-se de que models.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._models_data = data.get("models", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_models(self._models_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em models.json:\n{error_msg}")
        
        # Parsear modelos
        models = {}
        for m in self._models_data:
            model = ModelSpec(
                name=m["name"],
                num_layers=m["num_layers"],
                num_key_value_heads=m["num_key_value_heads"],
                head_dim=m["head_dim"],
                max_position_embeddings=m["max_position_embeddings"],
                attention_pattern=m["attention_pattern"],
                hybrid_full_layers=m.get("hybrid_full_layers"),
                hybrid_sliding_layers=m.get("hybrid_sliding_layers"),
                sliding_window=m.get("sliding_window"),
                default_kv_precision=m.get("default_kv_precision", "fp8"),
                total_params_b=m.get("total_params_b"),
                active_params_b=m.get("active_params_b"),
                weights_memory_gib_fp16=m.get("weights_memory_gib_fp16"),
                weights_memory_gib_bf16=m.get("weights_memory_gib_bf16"),
                weights_memory_gib_fp8=m.get("weights_memory_gib_fp8"),
                weights_memory_gib_int8=m.get("weights_memory_gib_int8"),
                weights_memory_gib_int4=m.get("weights_memory_gib_int4"),
                default_weights_precision=m.get("default_weights_precision", "fp8"),
                notes=m.get("notes", "")
            )
            model.validate()
            models[model.name.lower()] = model
        
        self._models = models
        return models
    
    def load_servers(self, filepath: str = "servers.json") -> Dict[str, ServerSpec]:
        """Carrega especificações de servidores do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Arquivo de servidores não encontrado: {path}\n"
                "Certifique-se de que servers.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._servers_data = data.get("servers", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_servers(self._servers_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em servers.json:\n{error_msg}")
        
        # Parsear servidores (estrutura nested)
        servers = {}
        for s in self._servers_data:
            server = self._parse_server(s)
            server.validate()
            servers[server.name.lower()] = server
        
        self._servers = servers
        return servers
    
    def _parse_server(self, s: Dict[str, Any]) -> ServerSpec:
        """Parse servidor com estrutura hierárquica."""
        
        # Parse CPU (opcional)
        cpu = None
        if "cpu" in s and s["cpu"]:
            cpu = CPUSpec(
                model=s["cpu"].get("model"),
                cores_total=s["cpu"].get("cores_total"),
                threads_total=s["cpu"].get("threads_total"),
                base_frequency_ghz=s["cpu"].get("base_frequency_ghz"),
                max_boost_frequency_ghz=s["cpu"].get("max_boost_frequency_ghz")
            )
        
        # Parse System Memory (opcional)
        system_memory = None
        if "system_memory" in s and s["system_memory"]:
            system_memory = SystemMemorySpec(
                capacity_total_tb=s["system_memory"].get("capacity_total_tb"),
                type=s["system_memory"].get("type"),
                speed_mhz=s["system_memory"].get("speed_mhz")
            )
        
        # Parse GPU (obrigatório)
        gpu = None
        if "gpu" in s and s["gpu"]:
            gpu = GPUSpec(
                count=s["gpu"]["count"],
                model=s["gpu"]["model"],
                hbm_per_gpu_gb=s["gpu"]["hbm_per_gpu_gb"],
                total_hbm_gb=s["gpu"].get("total_hbm_gb"),
                nvlink_bandwidth_tbps_total=s["gpu"].get("nvlink_bandwidth_tbps_total"),
                nvlink_generation=s["gpu"].get("nvlink_generation")
            )
        
        # Parse Power (obrigatório)
        power = None
        if "power" in s and s["power"]:
            power_supplies = None
            if "power_supplies" in s["power"] and s["power"]["power_supplies"]:
                power_supplies = PowerSupplySpec(
                    count=s["power"]["power_supplies"].get("count"),
                    rating_each_watts=s["power"]["power_supplies"].get("rating_each_watts"),
                    redundancy=s["power"]["power_supplies"].get("redundancy")
                )
            
            max_current = None
            if "max_current" in s["power"] and s["power"]["max_current"]:
                max_current = MaxCurrentSpec(
                    _208v_3phase_amps=s["power"]["max_current"].get("208v_3phase_amps"),
                    _480v_3phase_amps=s["power"]["max_current"].get("480v_3phase_amps")
                )
            
            power = PowerSpec(
                power_kw_max=s["power"]["power_kw_max"],
                power_supplies=power_supplies,
                input_voltage=s["power"].get("input_voltage"),
                max_current=max_current
            )
        
        # Parse Thermal (opcional)
        thermal = None
        if "thermal" in s and s["thermal"]:
            thermal = ThermalSpec(
                heat_output_btu_hr_max=s["thermal"].get("heat_output_btu_hr_max"),
                ambient_temp_operating_c_min=s["thermal"].get("ambient_temp_operating_c_min"),
                ambient_temp_operating_c_max=s["thermal"].get("ambient_temp_operating_c_max")
            )
        
        # Parse Cooling (opcional)
        cooling = None
        if "cooling" in s and s["cooling"]:
            cooling = CoolingSpec(
                airflow_cfm=s["cooling"].get("airflow_cfm"),
                cooling_type=s["cooling"].get("cooling_type")
            )
        
        # Parse Storage (opcional)
        storage_spec = None
        if "storage" in s and s["storage"]:
            storage_spec = StorageSpec(
                boot_drives=s["storage"].get("boot_drives"),
                internal_nvme_slots=s["storage"].get("internal_nvme_slots"),
                max_internal_storage_tb=s["storage"].get("max_internal_storage_tb")
            )
        
        # Parse Networking (opcional)
        networking = None
        if "networking" in s and s["networking"]:
            networking = NetworkingSpec(
                infiniband=s["networking"].get("infiniband"),
                management=s["networking"].get("management")
            )
        
        # Parse Software (opcional)
        software = None
        if "software" in s and s["software"]:
            software = SoftwareSpec(
                os_supported=s["software"].get("os_supported"),
                nvidia_ai_enterprise=s["software"].get("nvidia_ai_enterprise"),
                cuda_version=s["software"].get("cuda_version")
            )
        
        # Parse Physical (opcional)
        physical = None
        if "physical" in s and s["physical"]:
            dimensions = None
            if "dimensions_mm" in s["physical"] and s["physical"]["dimensions_mm"]:
                dimensions = DimensionsSpec(
                    width=s["physical"]["dimensions_mm"].get("width"),
                    depth=s["physical"]["dimensions_mm"].get("depth"),
                    height=s["physical"]["dimensions_mm"].get("height")
                )
            
            physical = PhysicalSpec(
                dimensions_mm=dimensions,
                weight_kg_max=s["physical"].get("weight_kg_max")
            )
        
        # Criar ServerSpec
        return ServerSpec(
            name=s["name"],
            manufacturer=s.get("manufacturer"),
            form_factor=s.get("form_factor"),
            rack_units_u=s.get("rack_units_u", 10),
            cpu=cpu,
            system_memory=system_memory,
            gpu=gpu,
            power=power,
            thermal=thermal,
            cooling=cooling,
            storage=storage_spec,
            networking=networking,
            software=software,
            physical=physical,
            notes=s.get("notes", ""),
            source=s.get("source")
        )
    
    def load_storage(self, filepath: str = "storage.json") -> Dict[str, StorageProfile]:
        """Carrega perfis de storage do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Arquivo de storage não encontrado: {path}\n"
                "Certifique-se de que storage.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._storage_data = data.get("profiles", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_storage_profiles(self._storage_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em storage.json:\n{error_msg}")
        
        # Parsear perfis de storage
        profiles = {}
        for p in self._storage_data:
            # Retrocompatibilidade: converter Gbps para MB/s se necessário
            throughput_read_mbps = p.get("throughput_read_mbps", 0.0)
            throughput_write_mbps = p.get("throughput_write_mbps", 0.0)
            
            # Se não tem MB/s mas tem Gbps (formato antigo), converter
            if throughput_read_mbps == 0.0 and "throughput_read_gbps" in p:
                throughput_read_mbps = p["throughput_read_gbps"] * 125.0  # Gbps → MB/s
            if throughput_write_mbps == 0.0 and "throughput_write_gbps" in p:
                throughput_write_mbps = p["throughput_write_gbps"] * 125.0
            
            profile = StorageProfile(
                name=p["name"],
                type=p["type"],
                capacity_total_tb=p.get("capacity_total_tb", 0.0),
                usable_capacity_tb=p.get("usable_capacity_tb", 0.0),
                iops_read_max=p.get("iops_read_max", 0),
                iops_write_max=p.get("iops_write_max", 0),
                throughput_read_mbps=throughput_read_mbps,
                throughput_write_mbps=throughput_write_mbps,
                block_size_kb_read=p.get("block_size_kb_read", 0.0),
                block_size_kb_write=p.get("block_size_kb_write", 0.0),
                latency_read_ms_p50=p.get("latency_read_ms_p50", 0.0),
                latency_read_ms_p99=p.get("latency_read_ms_p99", 0.0),
                latency_write_ms_p50=p.get("latency_write_ms_p50", 0.0),
                latency_write_ms_p99=p.get("latency_write_ms_p99", 0.0),
                rack_units_u=p.get("rack_units_u", 0),
                power_kw=p.get("power_kw", 0.0),
                notes=p.get("notes", "")
            )
            profile.validate()
            profiles[profile.name.lower()] = profile
        
        self._storage = profiles
        return profiles
    
    def get_model(self, name: str) -> ModelSpec:
        """Busca modelo por nome (case-insensitive)."""
        if not self._models:
            self.load_models()
        
        name_normalized = name.lower()
        if name_normalized not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(
                f"❌ Modelo '{name}' não encontrado em models.json.\n"
                f"Modelos disponíveis: {available}"
            )
        return self._models[name_normalized]
    
    def get_server(self, name: str) -> ServerSpec:
        """Busca servidor por nome (case-insensitive)."""
        if not self._servers:
            self.load_servers()
        
        name_normalized = name.lower()
        if name_normalized not in self._servers:
            available = ", ".join(self._servers.keys())
            raise ValueError(
                f"❌ Servidor '{name}' não encontrado em servers.json.\n"
                f"Servidores disponíveis: {available}"
            )
        return self._servers[name_normalized]
    
    def get_storage(self, name: str) -> StorageProfile:
        """Busca perfil de storage por nome (case-insensitive)."""
        if not self._storage:
            self.load_storage()
        
        name_normalized = name.lower()
        if name_normalized not in self._storage:
            available = ", ".join(self._storage.keys())
            raise ValueError(
                f"❌ Perfil de storage '{name}' não encontrado em storage.json.\n"
                f"Perfis disponíveis: {available}"
            )
        return self._storage[name_normalized]
    
    def get_raw_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retorna dados brutos (não parseados) para validação.
        
        Returns:
            (models_data, servers_data, storage_data)
        """
        return self._models_data, self._servers_data, self._storage_data
