"""
Cálculos de consumo físico de datacenter (energia, rack, dissipação térmica).
"""

from .calc_scenarios import ScenarioResult
from .servers import ServerSpec


# Constantes de conversão
WATTS_TO_BTU_HR = 3.412142  # 1 Watt = 3.412142 BTU/hr


def calc_physical_consumption(
    scenario: ScenarioResult,
    server: ServerSpec
) -> None:
    """
    Calcula consumo físico de datacenter para um cenário.
    
    Atualiza o scenario in-place com:
    - total_power_kw
    - total_rack_u
    - total_heat_btu_hr
    
    Args:
        scenario: Resultado do cenário a ser atualizado
        server: Especificação do servidor
    """
    nodes_final = scenario.nodes_final
    
    # Energia total (kW)
    if server.power and server.power.power_kw_max is not None:
        scenario.total_power_kw = nodes_final * server.power.power_kw_max
    else:
        scenario.total_power_kw = 0.0
    
    # Rack space total (U)
    if server.rack_units_u is not None:
        scenario.total_rack_u = nodes_final * server.rack_units_u
    else:
        scenario.total_rack_u = 0
    
    # Dissipação térmica (BTU/hr)
    if scenario.total_power_kw > 0:
        total_watts = scenario.total_power_kw * 1000
        scenario.total_heat_btu_hr = total_watts * WATTS_TO_BTU_HR
    else:
        scenario.total_heat_btu_hr = 0.0
