import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def fig2data(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    return data.reshape((height, width, 3))

def MainApp():
    st.title("Análisis de experimentos para parametrización")
    material=st.text_input("Titulo del experimento")

    # Factor 1
    st.markdown("## Factor 1")
    namFactor1 = st.text_input("Inserta el nombre del factor 1", "Factor 1")
    Factor1 = st.number_input("Inserta el factor 1 recomendado", 10, 400, 200)
    Varianza1 = st.number_input("Inserta la varianza del factor 1", 5, 40, 20)
    minF1, maxF1 = Factor1 - Varianza1, Factor1 + Varianza1

    # Factor 2
    st.markdown("## Factor 2")
    namFactor2 = st.text_input("Inserta el nombre del factor 2", "Factor 2")
    Factor2 = st.number_input("Inserta el factor 2 recomendado", 10, 400, 200)
    Varianza2 = st.number_input("Inserta la varianza del factor 2", 5, 40, 20)
    minF2, maxF2 = Factor2 - Varianza2, Factor2 + Varianza2

    # Generate grid points for plotting`
    st.markdown("## Experimento")
    x1 = np.linspace(minF1, maxF1, 3)  # Values of factor 1
    x2 = np.linspace(minF2, maxF2, 3)  # Values of factor 2
    X_1, X_2 = np.meshgrid(x1, x2)

    # Plot the grid using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(X_1.ravel(), X_2.ravel(), marker='o', linestyle='', color='r')
    ax.set_xlabel(namFactor1)
    ax.set_ylabel(namFactor2)
    ax.set_title('Gráfica de experimento')

    # Convert the Matplotlib figure to an image
    fig_image = fig2data(fig)

    # Display the image in Streamlit
    st.image(fig_image, use_column_width=True)

    # Collect evaluation values
    eval_inputs = []
    for i in range(9):
        x1_val, x2_val = X_1.ravel()[i], X_2.ravel()[i]
        eval_val = st.number_input(f"Evaluación de {x1_val},{x2_val}", 1, 10, 1)
        eval_inputs.append(eval_val)

    y = np.array(eval_inputs)  # Response variable values

    # Fit a quadratic model to the data
    modeloDeX = np.column_stack((np.ones((1, 9)).ravel(), X_1.ravel(), X_2.ravel(), X_1.ravel()**2, X_2.ravel()**2, X_1.ravel()*X_2.ravel()))
    coeffs = np.linalg.lstsq(modeloDeX, y, rcond=None)[0]

    # Define the response function
    def response_function(x):
        return coeffs[0] + coeffs[1]*x[0] + coeffs[2]*x[1] + coeffs[3]*x[0]**2 + coeffs[4]*x[1]**2 + coeffs[5]*x[0]*x[1]

    # Optimize the response function
    initial_guess = [Factor1, Factor2]  # Initial guess for factor settings
    result = minimize(response_function, initial_guess, bounds=[(minF1, maxF1), (minF2, maxF2)])

    # Generate grid points for plotting the surface response
    x1_plot = np.linspace(minF1, maxF1, 100)
    x2_plot = np.linspace(minF2, maxF2, 100)
    X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)
    Z_plot = response_function([X1_plot, X2_plot])

    # Find the pair of values that optimize the response
    optimal_indices = np.unravel_index(np.argmax(Z_plot), Z_plot.shape)
    optimal_values = [X1_plot[optimal_indices], X2_plot[optimal_indices]]
    optimal_response = Z_plot[optimal_indices]

    # Plot the surface response
    fig_response = plt.figure()
    ax_response = fig_response.add_subplot(111, projection='3d')
    ax_response.plot_surface(X1_plot, X2_plot, Z_plot, cmap='viridis')
    ax_response.set_xlabel(namFactor1)
    ax_response.set_ylabel(namFactor2)
    ax_response.set_zlabel('Response')
    ax_response.set_title('Superficie de respuesta')

    # Convert the Matplotlib figure to an image
    fig_image_response = fig2data(fig_response)

    # Display the image in Streamlit
    st.image(fig_image_response, use_column_width=True)

    # Plot the contour graph
    fig_contour, ax_contour = plt.subplots()
    contour_plot = ax_contour.contourf(X1_plot, X2_plot, Z_plot, levels=20, cmap='viridis')
    ax_contour.set_xlabel(namFactor1)
    ax_contour.set_ylabel(namFactor2)
    ax_contour.set_title('Gráfico de contorno')

    # Add color values to the contour plot
    contour_labels = ax_contour.contour(X1_plot, X2_plot, Z_plot, colors='k')
    ax_contour.clabel(contour_labels, inline=True, fontsize=8)

    # Add a colorbar
    colorbar = plt.colorbar(contour_plot)
    colorbar.set_label('Response')

    # Convert the Matplotlib figure to an image
    fig_image_contour = fig2data(fig_contour)

    # Display the image in Streamlit
    st.image(fig_image_contour, use_column_width=True)

    # Print the optimal values
    st.write("### Optimal Values:")
    st.write(f"{namFactor1}: {optimal_values[0]}")
    st.write(f"{namFactor2}: {optimal_values[1]}")
    st.write("Optimal Response:", optimal_response)

MainApp()
