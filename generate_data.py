# generate_data.py
import numpy as np
import pandas as pd

CROP_PROFILES = {
    "Spinach": {"ideal_moisture": 60, "ideal_light": 8},
    "Lettuce": {"ideal_moisture": 55, "ideal_light": 10},
    "Tomato": {"ideal_moisture": 65, "ideal_light": 12},
    "Coriander": {"ideal_moisture": 50, "ideal_light": 9},
    "Wheat": {"ideal_moisture": 70, "ideal_light": 7}
}

def generate_single_crop(crop_name="Spinach", days=60, seed=None):
    if seed is not None:
        np.random.seed(seed)
    profile = CROP_PROFILES[crop_name]
    days_idx = np.arange(days)

    # Simulate natural fluctuations around an ideal
    moisture = np.clip(
        np.random.normal(loc=profile["ideal_moisture"], scale=8, size=days).astype(int),
        10, 99
    )
    nutrients = np.clip(np.random.normal(loc=70, scale=10, size=days).astype(int), 10, 120)
    light_hours = np.clip(np.random.normal(loc=profile["ideal_light"], scale=1.5, size=days).round(1), 1, 18)
    temperature = np.clip(np.random.normal(loc=27, scale=3.0, size=days).round(1), 10, 45)

    # growth_index: synthetic target (0..100)
    # depends positively on moisture close to ideal, nutrients, light close to ideal, and temperature in comfortable range
    moisture_effect = 1 - np.abs(moisture - profile["ideal_moisture"]) / 50
    light_effect = 1 - np.abs(light_hours - profile["ideal_light"]) / 10
    temp_effect = 1 - (np.abs(temperature - 25) / 20)
    growth_index = (0.5 * moisture_effect + 0.3 * (nutrients / 100) + 0.2 * light_effect + 0.1 * temp_effect)
    # Normalize into 0-100
    growth_index = np.clip((growth_index / growth_index.max()) * 100, 0, 100).round(1)

    # irrigation label: if moisture < ideal - 10 -> 'irrigate_now', elif moisture < ideal -> 'soon', else 'ok'
    irrigation = []
    for m in moisture:
        if m < profile["ideal_moisture"] - 10:
            irrigation.append("irrigate_now")
        elif m < profile["ideal_moisture"]:
            irrigation.append("soon")
        else:
            irrigation.append("ok")

    # stage (seedling, vegetative, harvest) - simple mapping by day
    stage = []
    for d in days_idx:
        if d < days * 0.25:
            stage.append("seedling")
        elif d < days * 0.75:
            stage.append("vegetative")
        else:
            stage.append("harvest")

    df = pd.DataFrame({
        "day": days_idx,
        "crop": crop_name,
        "moisture": moisture,
        "nutrients": nutrients,
        "light_hours": light_hours,
        "temperature": temperature,
        "growth_index": growth_index,
        "irrigation_label": irrigation,
        "stage": stage
    })
    return df

def generate_full_dataset(days=60, seed=None):
    # Create dataset by concatenating for each crop
    dfs = []
    for crop in CROP_PROFILES.keys():
        dfs.append(generate_single_crop(crop, days=days, seed=(None if seed is None else seed + hash(crop) % 1000)))
    full = pd.concat(dfs, ignore_index=True)
    return full

if __name__ == "__main__":
    df = generate_full_dataset(days=60, seed=42)
    df.to_csv("simulated_farm_data.csv", index=False)
    print("Wrote simulated_farm_data.csv (shape: {})".format(df.shape))
