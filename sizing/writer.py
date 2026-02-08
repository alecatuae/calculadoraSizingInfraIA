"""
Writer: escreve relatórios em arquivos (txt, json).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ReportWriter:
    """Gerencia escrita de relatórios em ./relatorios."""
    
    def __init__(self, base_dir: str = "relatorios"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def _generate_filename(self, model_name: str, server_name: str, extension: str) -> Path:
        """Gera nome de arquivo com timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sizing_{model_name}_{server_name}_{timestamp}.{extension}"
        return self.base_dir / filename
    
    def write_text_report(
        self,
        content: str,
        model_name: str,
        server_name: str
    ) -> Path:
        """Escreve relatório completo em texto."""
        filepath = self._generate_filename(model_name, server_name, "txt")
        filepath.write_text(content, encoding='utf-8')
        return filepath
    
    def write_json_report(
        self,
        data: Dict[str, Any],
        model_name: str,
        server_name: str
    ) -> Path:
        """Escreve relatório completo em JSON."""
        filepath = self._generate_filename(model_name, server_name, "json")
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        return filepath
    
    def write_executive_report(
        self,
        content: str,
        model_name: str,
        server_name: str
    ) -> Path:
        """Escreve relatório executivo em Markdown."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"executive_{model_name}_{server_name}_{timestamp}.md"
        filepath = self.base_dir / filename
        filepath.write_text(content, encoding='utf-8')
        return filepath
