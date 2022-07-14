## CPT Code Assignment Review Application

### Purpose:

This dashboard is designed to display CPT codes that may have been missed by the original coders. It does this by highlighting reports where the coders' assignments are not the same as the assignments from our XGBoost model.

### Opening app:

1. Long onto Discovery on whichever open port you can and then access a compute node.
2. Before you pip install I recommended you create a new Conda environment to ensure everything works:

    ```bash
    conda create --name <enviro name>
    conda activate <enviro name>
    conda install git pip  # ~6 minutes
    ```

3. Install package:
    ```bash
    pip install git+https://github.com/jackgreenburg/cpt_code_app.git  # ~20 minutes
    ```
4. Start app:
    ```bash
    cpt-code-app --port=<your current port>
    ```
5. Open `localhost:<port>` in a web browser

### Operation:

Feel free to test whatever you like. The app is geared towards identifying potential underbilling, so that would mean applying the filters that limit the app to only display false negatives.
