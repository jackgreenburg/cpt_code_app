## CPT Code Assignment Review Application

### Purpose:

This dashboard is designed to display CPT codes that may have been missed by the original coders. It does this by highlighting reports where the coders' assignments are not the same as the assignments from our XGBoost model.

### Opening app:

1. Log onto Discovery on whichever open port you can and then access a compute node (I always use gv01 and haven't had any problems yet). **It unfortunately cannot run locally because it needs to access files that are on Discovery.**
2. Before you pip install I recommended you create a new Conda environment to ensure everything works:

    ```bash
    conda create --name <enviro name>
    conda activate <enviro name>
    conda install git pip  # ~6 minutes
    ```

3. Install package (you probably will need to login with your username and a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)):
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
