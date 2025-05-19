from flask import Flask, render_template, request, session, flash, redirect, url_for, jsonify
import pandas as pd
import os
import markdown
from src.agent.diabetes_agent import DiabetesAgent 
from src import config as AppConfig  
import json

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_super_secret_random_string_here'  # IMPORTANT!

    app.config['UPLOAD_FOLDER'] = os.path.join(file_dir, 'uploads')
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    csv_path = 'data/diabetic_data.csv'  # Ajusta la ruta a tu CSV

    # Cargar datos al iniciar la app
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    @app.route('/')
    def titulo():
        return render_template('titulo.html')
    

    @app.route('/results', methods=['GET']) # New dedicated route for results
    def display_diagnosis_results():
        # Retrieve data from session
        explanation_markdown = session.get('explanation_markdown')
        plot_filename = session.get('plot_filename')

        if not explanation_markdown or not plot_filename:
            # Handle case where data is not in session (e.g., direct access to /results-page)
            return "No diagnosis data found. Please upload a file first.", 400 # Or redirect to upload page

        # Convert Markdown to HTML
        html_explanation = markdown.markdown(explanation_markdown, extensions=['fenced_code', 'tables', 'extra'])
        
        # Construct the URL for the plot image using url_for('static', ...)
        # This assumes 'plot_filename' is the path *within* your static folder.
        # E.g., if static folder is 'static/' and plot_filename is 'plots/myplot.png',
        # then the actual file is at 'static/plots/myplot.png'.
        actual_plot_url = url_for('static', filename=plot_filename)

        # Render the standalone results page
        return render_template('results.html', 
                            explanation=html_explanation, 
                            plot_url=actual_plot_url)
    
    @app.route('/diagnosis-questionnaire')
    def questionnaire():
        return render_template('dinamic_questionnaire.html')  # Página del cuestionario
    
    @app.route('/search/icd9')
    def search_icd9():
        with open('data/icd9Hierarchy.json') as f:
            icd9_data = json.load(f)
        query = request.args.get('q', '')
        results = []
        for item in icd9_data:
            if "subchapter" in item.keys():
                description = item["major"] + " " + item["subchapter"] + " " + item["chapter"]
            else:
                description = item["major"] + " " + item["chapter"]

            if item['icd9'].startswith(query):
                results.append({"numero": item["icd9"], "desc": item["descLong"]})

            if item["threedigit"].startswith(query):
                results.append({"numero": item["threedigit"], "desc": description})
            
            if not query.lower().isdigit():
                if query.lower() in item["descLong"].lower():
                    results.append({"numero": item["icd9"], "desc": item["descLong"]})
                
                if query.lower() in description.lower():
                    results.append({"numero": item["threedigit"], "desc": description})
        return jsonify(results)
    
    # @app.route('/search/icd9/<icd9>')
    # def search_icd9_by_code(icd9):
    #     with open('data/icd9Hierarchy.json') as f:
    #         icd9_data = json.load(f)
    #     for item in icd9_data:
    #         if item['icd9'] == icd9:
    #             return jsonify({"numero":item["icd9"], "desc": item["descLong"]})
    #         elif item["threedigit"] == icd9:
    #             return jsonify({"numero":item["threedigit"], "desc": item["major"]+" "+item["subchapter"]+" "+item["chapter"]})
    #     return jsonify({"error": "ICD-9 code not found"}), 404

    @app.route('/search-patient')
    def search_patient():
        query = request.args.get('q', '')
        df = pd.read_csv(csv_path)
        # Convertir NaN a None y manejar tipos de datos
        results = df[df['patient_nbr'].astype(str).str.contains(query)].head(5)
        clean_results = results.where(pd.notnull(results), None).to_dict(orient='records')
        
        return jsonify(clean_results)

    @app.route('/get-patient-data/<patient_id>')
    def get_patient_data(patient_id):
        df = pd.read_csv(csv_path)
        patient_data = df[df['patient_nbr'].astype(str) == str(patient_id)]
        
        if patient_data.empty:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Limpiar datos antes de convertir a JSON
        clean_data = patient_data.iloc[0].replace({pd.NA: None, pd.NaT: None}).to_dict()
        
        return jsonify(clean_data)
    ##################### modo CSV #####################

    @app.route('/upload-csv', methods=['GET'])
    def upload_csv():
        return render_template('upload_csv.html')

    @app.route('/load', methods=['POST'])
    def load():
        if 'csvFile' not in request.files:
            return "ERROR: No se seleccionó ningún archivo.", 400

        file = request.files['csvFile']

        if file.filename == '':
            return "ERROR: El archivo no tiene nombre.", 400

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Guarda el archivo en el servidor
            agent = DiabetesAgent(cfg=AppConfig)
            full_analysis_output = agent.generate_diagnostic_explanation('uploads/latent_data.csv')
            explanation=full_analysis_output['llm_generated_explanation']
            #html_explanation = markdown.markdown(explanation, extensions=['fenced_code', 'tables', 'extra'])
            
            # # Ruta al archivo PNG en la carpeta `static`
            plot_url = 'results/shap_explanations/patient_37096866_summary_bar.png'

            # # Explicación de ejemplo
            # explanation = l
            # Save the explanation to a file in the uploads directory
            explain_path = os.path.join(app.config['UPLOAD_FOLDER'], 'explain.txt')
            with open(explain_path, 'w', encoding='utf-8') as f:
                f.write(explanation)
            #return render_template('results.html', plot_url=plot_url, explanation=html_explanation)
            session['explanation_markdown'] = explanation  # Store the explanation in session   
            session['plot_filename'] = plot_url # Store the filename

            # Redirect to the dedicated results display page
            return redirect(url_for('display_diagnosis_results'))
                
        else:
            return "ERROR: El archivo debe ser un CSV.", 400
        

    ####################### modo cuestionario #######################

    @app.route('/submit-questionnaire', methods=['POST'])
    def submit_questionnaire():
        # Obtener los datos enviados por el formulario
        encounter_id = request.form.get('encounter_id', type=int)  # Encounter ID
        patient_id = request.form.get('patient_nbr', type=int)
        age = request.form.get('age', type=int)  # Age at Diagnosis
        race = request.form.get('race', type=str)
        gender = request.form.get('gender', type=str)
        admission_source_id = request.form.get('admission_source_id', type=int)
        admission_type_id = request.form.get('admission_type_id', type=int)
        discharge_disposition_id = request.form.get('discharge_disposition_id', type=int)
        time_in_hospital = request.form.get('time_in_hospital', type=int)
        num_lab_procedures = request.form.get('num_lab_procedures', type=int)
        num_procedures = request.form.get('num_procedures', type=int)
        num_medications = request.form.get('num_medications', type=int)
        number_outpatient = request.form.get('number_outpatient', type=int)
        number_emergency = request.form.get('number_emergency', type=int)
        number_inpatient = request.form.get('number_inpatient', type=int)
        number_diagnoses = request.form.get('number_diagnoses', type=int)
        metformin = request.form.get('metformin', type=str)
        repaglinide = request.form.get('repaglinide', type=str)
        nateglinide = request.form.get('nateglinide', type=str)
        chlorpropamide = request.form.get('chlorpropamide', type=str)
        glimepiride = request.form.get('glimepiride', type=str)
        acetohexamide = request.form.get('acetohexamide', type=str)
        miglitol = request.form.get('miglitol', type=str)
        troglitazone = request.form.get('troglitazone', type=str)
        tolazamide = request.form.get('tolazamide', type=str)
        glyburide_metformin = request.form.get('glyburide-metformin', type=str)
        glipizide_metformin = request.form.get('glipizide-metformin', type=str)
        glimepiride_pioglitazone = request.form.get('glimepiride-pioglitazone', type=str)
        metformin_rosiglitazone = request.form.get('metformin-rosiglitazone', type=str)
        metformin_pioglitazone = request.form.get('metformin-pioglitazone', type=str)
        acarbose = request.form.get('acarbose', type=str)
        insulin = request.form.get('insulin', type=str)
        glyburide = request.form.get('glyburide', type=str)
        pioglitazone = request.form.get('pioglitazone', type=str)
        examide = request.form.get('examide', type=str)
        citoglipton = request.form.get('citoglipton', type=str)
        glipizide = request.form.get('glipizide', type=str)
        tolbutamide = request.form.get('tolbutamide', type=str)
        rosiglitazone = request.form.get('rosiglitazone', type=str)
        diag_1 = request.form.get('diag_1', type=str)
        diag_2 = request.form.get('diag_2', type=str)
        diag_3 = request.form.get('diag_3', type=str)

        age_mapper = {0: '[0-10)', 1: '[10-20)', 2: '[20-30)', 3: '[30-40)', 4: '[40-50)', 5: '[50-60)', 6: '[60-70)', 7: '[70-80)', 8: '[80-90)', 9: '[90-100)'}
        # Guardar los datos en variables o procesarlos
        user_data = {
            'encounter_id': encounter_id,
            'patient_nbr': patient_id,
            'race': race,
            'gender': gender,
            'age': age_mapper[age//10],
            'weight': '',
            'admission_type_id': admission_type_id,
            'discharge_disposition_id': discharge_disposition_id,
            'admission_source_id': admission_source_id,
            'time_in_hospital': time_in_hospital,
            'payer_code': '',
            'medical_specialty': '',
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'diag_1': diag_1 if diag_1 != '' else '?',
            'diag_2': diag_2 if diag_2 != '' else '?',
            'diag_3': diag_3 if diag_3 != '' else '?',
            'number_diagnoses': number_diagnoses,
            'max_glu_serum': '',
            'A1Cresult': '',
            'metformin': metformin,
            'repaglinide': repaglinide,
            'nateglinide': nateglinide,
            'chlorpropamide': chlorpropamide,
            'glimepiride': glimepiride,
            'acetohexamide': acetohexamide,
            'glipizide': glipizide,
            'glyburide': glyburide,
            'tolbutamide': tolbutamide,
            'pioglitazone': pioglitazone,
            'rosiglitazone': rosiglitazone,
            'acarbose': acarbose,
            'miglitol': miglitol,
            'troglitazone': troglitazone,
            'tolazamide': tolazamide,
            'examide': examide,
            'citoglipton': citoglipton,
            'insulin': insulin,
            'glyburide-metformin': glyburide_metformin,
            'glipizide-metformin': glipizide_metformin,
            'glimepiride-pioglitazone': glimepiride_pioglitazone,
            'metformin-rosiglitazone': metformin_rosiglitazone,
            'metformin-pioglitazone': metformin_pioglitazone,
            'change': '',
            'diabetesMed': '',
            'readmitted': ''
        }
        User = pd.DataFrame(user_data, index=[0])
        User.to_csv('uploads/latent_data.csv', index=False)
        # print(User)
        agent = DiabetesAgent(cfg=AppConfig)
        full_analysis_output = agent.generate_diagnostic_explanation('uploads/latent_data.csv')
        explanation=full_analysis_output['llm_generated_explanation']
        html_explanation = markdown.markdown(explanation, extensions=['fenced_code', 'tables', 'extra'])

        plot_url = 'results/shap_explanations/patient_37096866_summary_bar.png'

        # Explicación de ejemplo
        
        return render_template('results.html', plot_url=plot_url, explanation=html_explanation)


    return app




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)