python -m venv venv
call venv\Scripts\activate
echo "Installing Libraries"
pip install -v trimesh
pip install -v Pillow
pip install -v panda
pip install -v numpy-stl
pip install -v pyvista
pip install -v numpy
pip install -v streamlit
pip install -v openpyxl
pip install -v python-dotenv
pip install -v meshio
pip install -v wget
pip install -v scipy

echo "Ready"
pause
