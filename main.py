import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import zipfile
import tempfile
import shutil
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Visualizador de Secciones Sistemáticas")


with st.sidebar:
    st.header("Parámetros de sección")
    origin_x = st.number_input("Origin X", value=394500)
    origin_y = st.number_input("Origin Y", value=8553300)
    azimuth = st.number_input("Azimuth", value=218)
    spacing = st.number_input("Spacing (m)", value=300)
    num_sections = st.number_input("Número de secciones", value=8, min_value=1, max_value=100)
    clip_width = st.number_input("Ancho de corte (m)", value=5)


st.subheader("Carga de archivos CSV")
main_file = st.file_uploader("Archivo principal", type=["csv"])
secondary_file = st.file_uploader("Archivo secundario", type=["csv"])
dh_file = st.file_uploader("Archivo de sondajes", type=["csv"])

# Ejecutar procesamiento si todos los archivos están cargados
if main_file and secondary_file and dh_file:

    df_main = pd.read_csv(main_file, low_memory=False)
    df_secondary = pd.read_csv(secondary_file, low_memory=False)
    df_dh = pd.read_csv(dh_file, encoding='latin1', low_memory=False)

    color_map = {2: 'Orange', 3: 'Green', 5: 'skyblue', 7: 'pink'}
    label_map = {2: 'Ganancia', 3: 'Conversión', 5: 'Económico', 7: 'UpNSR'}
    marker_map = {2: 's', 3: '^', 5: 'o', 7: 'v'}

    def nsr_color(nsr):
        if nsr >= 90:
            return '#800080'
        elif nsr >= 60:
            return '#FF0000'
        elif nsr >= 42:
            return '#FF8C00'
        elif nsr >= 30:
            return '#FFD700'
        elif nsr >= 15:
            return '#FFFF00'
        else:
            return '#A9A9A9'

    df_main = df_main[df_main['OREBODY'].str.strip().str.lower() != 'dique']
    df_main = df_main[df_main['TOPE'].isin([2, 3, 5, 7])]
    df_secondary = df_secondary[df_secondary['CGEOCD'] != 3]

    perp_azimuth = (azimuth + 90) % 360
    centers = []
    for i in range(int(num_sections)):
        dx = i * spacing * np.sin(np.radians(perp_azimuth))
        dy = i * spacing * np.cos(np.radians(perp_azimuth))
        centers.append((origin_x + dx, origin_y + dy))

    image_paths = []
    temp_dir = tempfile.mkdtemp()

    for idx, (cx, cy) in enumerate(centers):
        vx_clip = np.sin(np.radians(azimuth + 90))
        vy_clip = np.cos(np.radians(azimuth + 90))
        vx_proj = np.sin(np.radians(azimuth + 180) % 360)
        vy_proj = np.cos(np.radians(azimuth + 180) % 360)

        def within_clip(df, xcol, ycol):
            dx = df[xcol] - cx
            dy = df[ycol] - cy
            return np.abs(dx * vx_clip + dy * vy_clip) <= clip_width

        def project(df, xcol, ycol):
            dx = df[xcol] - cx
            dy = df[ycol] - cy
            return dx * vx_proj + dy * vy_proj

        section_main = df_main[within_clip(df_main, 'XC', 'YC')].copy()
        section_secondary = df_secondary[within_clip(df_secondary, 'XC', 'YC')].copy()
        section_dh = df_dh[within_clip(df_dh, 'X', 'Y')].copy()

        section_main['Y_proj'] = project(section_main, 'XC', 'YC')
        section_secondary['Y_proj'] = project(section_secondary, 'XC', 'YC')
        section_dh['Y_proj'] = project(section_dh, 'X', 'Y')

        section_main['NSR_color'] = section_main['NSR24RES'].apply(nsr_color)

        fig, axs = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[0.8, 1.5])

        axs[0].scatter(df_main['XC'], df_main['YC'], color='lightgray', s=2, alpha=0.5)
        axs[0].plot([cx - 1500 * vx_proj, cx + 1500 * vx_proj],
                    [cy - 1500 * vy_proj, cy + 1500 * vy_proj],
                    color='red', linewidth=2, label='Ubicación sección')

        for orebody in section_main['OREBODY'].dropna().unique():
            sub = section_main[section_main['OREBODY'] == orebody]
            axs[0].text(sub['XC'].median(), sub['YC'].median(), orebody, fontsize=8,
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))

        axs[0].set_title(f"Vista en planta - Sección {idx+1}")
        axs[0].set_xlabel("Este (m)")
        axs[0].set_ylabel("Norte (m)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].scatter(section_secondary['Y_proj'], section_secondary['ZC'], color='lightgray', s=3, alpha=0.5)

        for tope in [2, 3, 5, 7]:
            group = section_main[section_main['TOPE'] == tope]
            axs[1].scatter(
                group['Y_proj'], group['ZC'],
                c=group['NSR_color'],
                s=10,
                marker=marker_map[tope],
                edgecolors='black',
                linewidths=0.2,
                label=label_map[tope]
            )

        for cod, color, label in [(1, 'black', 'Sondajes COD=1'), (2, 'blue', 'Sondajes COD=2')]:
            sondajes = section_dh[section_dh['COD'] == cod]
            for i, bhid in enumerate(sondajes['BHID'].dropna().unique()):
                sub = sondajes[sondajes['BHID'] == bhid]
                trayecto = sub.sort_values('Z')
                axs[1].plot(trayecto['Y_proj'], trayecto['Z'], color=color, linewidth=0.9, alpha=0.5,
                            label=label if i == 0 else "")
                axs[1].scatter(trayecto['Y_proj'], trayecto['Z'], color=color, s=0.5, alpha=0.3)

        for orebody in section_main['OREBODY'].dropna().unique():
            sub = section_main[section_main['OREBODY'] == orebody]
            axs[1].text(sub['Y_proj'].median() - 20, sub['ZC'].median(), orebody, fontsize=10,
                        ha='right', va='center', bbox=dict(facecolor='white', alpha=0.6))

        axs[1].legend(loc='upper right')
        axs[1].set_title(f"Vista en sección - Sección {idx+1} (Azimuth {azimuth}°)")
        axs[1].set_xlabel("Distancia sobre sección (m)")
        axs[1].set_ylabel("Elevación (m)")
        axs[1].grid(True)

        plt.tight_layout()
        filename = os.path.join(temp_dir, f"seccion_{idx+1:02}.jpg")
        fig.savefig(filename, dpi=300)
        image_paths.append(filename)
        plt.close()

    #Slider
    st.subheader("Vista de secciones")
    img_idx = st.slider("Selecciona sección", 1, len(image_paths), 1)
    st.image(image_paths[img_idx - 1], use_column_width=True)

    # ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for path in image_paths:
            zipf.write(path, arcname=os.path.basename(path))
    zip_buffer.seek(0)

    st.download_button("📦 Descargar ZIP con secciones", data=zip_buffer,
                       file_name="secciones.zip", mime="application/zip")

    # Limpieza opcional si se desea persistencia
    # shutil.rmtree(temp_dir)
else:
    st.info("Cargar los tres archivos para iniciar el procesamiento.")