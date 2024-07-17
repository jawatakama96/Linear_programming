import streamlit as st
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

def main():
    st.title("Linear Programming Calculator")

    goal = st.selectbox("Objective:", ["Maximize", "Minimize"])

    # Khai báo số lượng biến
    num_vars = st.number_input("Number of variables:", min_value=1, step=1, value=2)

    # Nhập hàm mục tiêu
    st.write("Objective Function:")
    obj_coeffs = []
    cols = st.columns(num_vars)
    for i, col in enumerate(cols):
        with col:
            coeff = st.text_input(f"X{i+1}", key=f"obj_coeff_{i}", value="")
            obj_coeffs.append(coeff)

    # Nhập số lượng ràng buộc
    num_constraints = st.number_input("Number of constraints:", min_value=1, step=1, value=1)

    constraints = []
    for i in range(num_constraints):
        st.write(f"Constraint {i+1}:")
        lhs_coeffs = []
        cols = st.columns(num_vars + 2)  # Thêm 2 cột cho dấu cộng và dấu so sánh
        for j, col in enumerate(cols[:num_vars]):
            with col:
                coeff = st.text_input(f"X{j+1} (C{i+1})", key=f"lhs_{i}_{j}", value="")
                lhs_coeffs.append(coeff)
        with cols[num_vars]:
            relation = st.selectbox("Relation", ["<=", ">=", "="], key=f"relation_{i}")
        with cols[num_vars + 1]:
            rhs = st.text_input(f"Right hand side (C{i+1})", key=f"rhs_{i}", value="")
        constraints.append((lhs_coeffs, relation, rhs))

    if st.button("Submit"):
        try:
            obj_coeffs = [float(coeff) for coeff in obj_coeffs]
            constraints = [(list(map(float, lhs)), relation, float(rhs)) for lhs, relation, rhs in constraints]
        except ValueError:
            st.error("Please enter valid numbers.")
            return

        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            st.error("Solver not created.")
            return

        # Tạo biến quyết định
        decision_vars = [solver.NumVar(0, solver.infinity(), f'x{i+1}') for i in range(num_vars)]

        # Hàm mục tiêu
        objective = solver.Objective()
        for i in range(num_vars):
            objective.SetCoefficient(decision_vars[i], obj_coeffs[i])
        
        if goal == "Maximize":
            objective.SetMaximization()
        else:
            objective.SetMinimization()

        # Thêm ràng buộc
        for lhs_coeffs, relation, rhs in constraints:
            if relation == "<=":
                constraint = solver.RowConstraint(-solver.infinity(), rhs, '')
            elif relation == ">=":
                constraint = solver.RowConstraint(rhs, solver.infinity(), '')
            elif relation == "=":
                constraint = solver.RowConstraint(rhs, rhs, '')

            for i in range(num_vars):
                constraint.SetCoefficient(decision_vars[i], lhs_coeffs[i])

        # Giải quyết bài toán
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            st.success(f"Solution found:")
            for i in range(num_vars):
                st.write(f"x{i+1} = {decision_vars[i].solution_value()}")
            st.write(f"Objective value = {solver.Objective().Value()}")
            
            # Tính tất cả các điểm cực biên và giá trị hàm mục tiêu
            vertices = find_vertices(constraints, num_vars)
            objective_values = []
            for vertex in vertices:
                value = sum(obj_coeffs[i] * vertex[i] for i in range(num_vars))
                objective_values.append((vertex, value))

            # Tạo dataframe để hiển thị bảng
            df = pd.DataFrame(objective_values, columns=['Coordinates', 'Objective Function Value'])
            df['Coordinates'] = df['Coordinates'].apply(lambda x: tuple(np.round(x, 2)))
            st.write(df)
        else:
            st.error("No optimal solution found.")

        # Vẽ đồ thị
        if num_vars == 2:
            x1_vals = np.linspace(0, 100, 400)
            fig, ax = plt.subplots()

            # Vẽ từng ràng buộc
            for lhs_coeffs, relation, rhs in constraints:
                if lhs_coeffs[1] != 0:  # Tránh chia cho 0
                    x2_vals = (rhs - lhs_coeffs[0] * x1_vals) / lhs_coeffs[1]
                    if relation == "<=":
                        ax.fill_between(x1_vals, x2_vals, 100, where=(x2_vals >= 0), alpha=0.3)
                    elif relation == ">=":
                        ax.fill_between(x1_vals, x2_vals, 0, where=(x2_vals <= 100), alpha=0.3)
                    ax.plot(x1_vals, x2_vals, label=f'{lhs_coeffs[0]}x1 + {lhs_coeffs[1]}x2 {relation} {rhs}')
                else:
                    ax.axvline(x=rhs/lhs_coeffs[0], label=f'{lhs_coeffs[0]}x1 {relation} {rhs}')

            ax.set_xlim((0, 100))
            ax.set_ylim((0, 100))
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.legend()
            plt.title("Feasible Region")
            st.pyplot(fig)

def find_vertices(constraints, num_vars):
    vertices = []
    for combo in combinations(constraints, num_vars):
        A = []
        b = []
        for (lhs_coeffs, relation, rhs) in combo:
            A.append(lhs_coeffs)
            b.append(rhs)
        try:
            vertex = np.linalg.solve(A, b)
            if all(vertex >= 0):  # Chỉ chấp nhận các điểm không âm
                valid = True
                for (lhs_coeffs, relation, rhs) in constraints:
                    if not evaluate_constraint(vertex, lhs_coeffs, relation, rhs):
                        valid = False
                        break
                if valid:
                    vertices.append(vertex)
        except np.linalg.LinAlgError:
            continue
    return vertices

def evaluate_constraint(point, lhs_coeffs, relation, rhs):
    value = sum(lhs_coeffs[i] * point[i] for i in range(len(lhs_coeffs)))
    if relation == "<=":
        return value <= rhs
    elif relation == ">=":
        return value >= rhs
    elif relation == "=":
        return value == rhs
    return False

if __name__ == "__main__":
    main()
