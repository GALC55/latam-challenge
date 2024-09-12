from locust import HttpUser, task
#tuve un error con jinja2 asi que actualice ese modulo y tambien el modulo de flask para poder correr el test de estres
class StressUser(HttpUser):
    
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas", 
                        "TIPOVUELO": "N", 
                        "MES": 3
                    }
                ]
            }
        )


    @task
    def predict_latam(self):
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM", 
                        "TIPOVUELO": "N", 
                        "MES": 3
                    }
                ]
            }
        )