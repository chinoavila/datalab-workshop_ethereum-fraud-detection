# Estructura del Proyecto de Detección de Fraude con Ethereum

Este documento detalla la organización y estructura completa del proyecto, explicando el propósito de cada directorio y archivo importante.

## Estructura General

```
datalab-workshop_ethereum-fraud-detection/
├── .vscode/
│   └── mcp.json                 # Configuración de Model Context Protocol
├── datasets/
│   └── transaction_dataset.csv  # Dataset principal de transacciones
├── labs/
│   └── model_x/
│       ├── model.py            # Implementación del modelo
│       ├── model_details.md    # Documentación detallada del modelo
│       └── requirements.txt     # Dependencias del modelo
├── notebooks/
│   └── model_x/
│       └── model.ipynb         # Notebook de desarrollo y análisis
├── notion/                      # Integración con Notion
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── LICENSE
│   ├── package.json
│   ├── tsconfig.json
│   ├── docs/
│   │   └── images/
│   │       ├── connections.png
│   │       ├── integrations-capabilities.png
│   │       └── integrations-creation.png
│   ├── scripts/
│   │   ├── build-cli.js
│   │   ├── notion-openapi.json
│   │   └── start-server.ts
│   └── src/
│       ├── init-server.ts
│       └── openapi-mcp-server/
└── README.md                    # Documentación principal del proyecto
```

## Descripción Detallada de los Componentes

### 1. Directorio Principal
- **README.md**: Documentación principal que proporciona una visión general del proyecto, instrucciones de instalación y uso.

### 2. Datos (.vscode/)
- **mcp.json**: Archivo de configuración para el Model Context Protocol, que gestiona la integración con diferentes servicios como Git, GitHub, y sistema de archivos.

### 3. Datasets (datasets/)
- **transaction_dataset.csv**: Contiene el conjunto de datos de transacciones de Ethereum utilizado para el análisis y entrenamiento de modelos.

### 4. Laboratorios (labs/)
En el directorio `model_x/`:
- **model.py**: Script principal que implementa el modelo de detección de fraude.
- **model_details.md**: Documentación técnica detallada sobre el modelo, incluyendo arquitectura y decisiones de diseño.
- **requirements.txt**: Lista de dependencias y bibliotecas necesarias para ejecutar el modelo.

### 5. Notebooks (notebooks/)
En el directorio `model_x/`:
- **model.ipynb**: Jupyter Notebook que contiene el desarrollo iterativo, análisis exploratorio de datos y experimentación con el modelo.

### 6. Integración con Notion (notion/)
Contiene la infraestructura para la integración con Notion:
- **docker-compose.yml**: Configuración de servicios Docker
- **Dockerfile**: Instrucciones para construir la imagen Docker
- **package.json**: Dependencias y scripts de Node.js
- **tsconfig.json**: Configuración de TypeScript

#### Subdirectorios de Notion:
- **docs/images/**: Recursos visuales para la documentación
- **scripts/**: Scripts de utilidad y configuración
- **src/**: Código fuente de la integración con Notion

## Uso de la Estructura

1. Los desarrolladores deben comenzar revisando el `README.md` principal para la configuración inicial.
2. El desarrollo de modelos se realiza principalmente en la carpeta `notebooks/`.
3. Los modelos finalizados se implementan en la carpeta `labs/`.
4. Los datos se mantienen centralizados en `datasets/`.
5. La integración con Notion proporciona una capa adicional de documentación y colaboración.

## Convenciones de Nomenclatura

- Los nombres de archivos utilizan snake_case (ejemplo: `model_details.md`)
- Los notebooks y scripts de modelo comparten el mismo nombre base para facilitar su relación
- La documentación se mantiene en archivos Markdown (.md) para mejor legibilidad

Esta estructura está diseñada para mantener una clara separación de responsabilidades y facilitar tanto el desarrollo como el mantenimiento del proyecto.