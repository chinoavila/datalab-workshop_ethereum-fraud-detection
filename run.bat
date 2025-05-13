@echo off
echo Activando entorno virtual...
call venv\Scripts\activate
echo Actualizando dependencias...
pip install --upgrade pip
pip install -r requirements.txt
echo Ejecutando aplicacion...
cd src
python -m streamlit run app.py
pause
