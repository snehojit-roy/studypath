import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import org.json.JSONObject;

public class Controller {

    @FXML
    private TextField cityField;

    @FXML
    private Label aqiLabel, statusLabel, pollutionLabel;

    private final String API_KEY = "YOUR_API_KEY";

    @FXML
    public void getAQI() {
        try {
            String city = cityField.getText();

            // Step 1: Get coordinates
            String geoUrl = "http://api.openweathermap.org/geo/1.0/direct?q=" 
                            + city + "&limit=1&appid=" + API_KEY;

            JSONObject geoData = new JSONObject(readUrl(geoUrl)).getJSONArray("results").getJSONObject(0);

            double lat = geoData.getDouble("lat");
            double lon = geoData.getDouble("lon");

            // Step 2: Get AQI
            String aqiUrl = "http://api.openweathermap.org/data/2.5/air_pollution?lat="
                            + lat + "&lon=" + lon + "&appid=" + API_KEY;

            JSONObject data = new JSONObject(readUrl(aqiUrl));
            int aqi = data.getJSONArray("list").getJSONObject(0).getJSONObject("main").getInt("aqi");

            updateUI(aqi);

        } catch (Exception e) {
            aqiLabel.setText("Error fetching AQI");
        }
    }

    private String readUrl(String urlString) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder result = new StringBuilder();

        String line;
        while ((line = reader.readLine()) != null) {
            result.append(line);
        }

        return result.toString();
    }

    private void updateUI(int aqi) {
        String status;

        switch (aqi) {
            case 1: status = "Good"; break;
            case 2: status = "Fair"; break;
            case 3: status = "Moderate"; break;
            case 4: status = "Poor"; break;
            case 5: status = "Very Poor"; break;
            default: status = "Unknown";
        }

        aqiLabel.setText("AQI: " + aqi);
        statusLabel.setText("Status: " + status);
    }
}