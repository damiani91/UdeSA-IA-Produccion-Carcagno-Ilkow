"""
Script para descargar los datasets de producción de pozos desde datos.gob.ar.
Código escrito por Gemini
"""
from pathlib import Path

import requests

# Variables globales
URLS = {
    "produccion": (
        "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/"
        "resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccion.csv"
    ),
    "pozos": (
        "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/"
        "resource/cbfa4d79-ffb3-4096-bab5-eb0dde9a8385/download/pozos.csv"
    ),
}

# Calculamos la ruta absoluta de la carpeta data/raw basándonos en la ubicación de este script
# Se resolverá exactamente a: /Users/.../UdeSA-IA-Produccion-Carcagno-Ilkow/data/raw
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"


def download_file(url: str, dest: Path) -> None:
    """Descarga un archivo desde una URL."""
    # Si la descarga automática falla, descargar manualmente y colocar en data/raw/.
    print(f"Descargando {url} → {dest}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f" ✓ {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Descargar o indicar descarga manual
    print("Iniciando descarga de datasets. Si la descarga automática falla, podés hacerlo manualmente desde:")
    for name, url in URLS.items():
        print(f" - {name}: {url}")
        print(f"   (Luego de descargar, colocar los CSVs en {RAW_DIR}/)")
        
        dest = RAW_DIR / f"{name}.csv"
        if dest.exists():
            print(f" ✓ El archivo {dest} ya existe. Omitiendo descarga.")
            continue
            
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Error descargando {name}: {e}")
