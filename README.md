# AI-Optimized Urban Farming System

## Overview
Urban farming simulation + AI-powered insights platform that integrates Indian organic principles (Panchgavya/Jeevamrut) with Groq's structured outputs API.

## Features

### 📊 Core Simulation
- **Multi-crop modeling**: Spinach, Lettuce, Tomato, Coriander, Wheat
- **Realistic data generation**: Improved correlations with temporal dependencies
- **ML models**: RandomForest for irrigation classification (99% accuracy) & growth regression (R² 0.7+)

### 🤖 AI-Powered Insights (Groq Integration)

1. **🩺 Crop Health Diagnosis**
   - Real-time health assessment with confidence scores
   - Risk factor identification (moisture stress, nutrient deficiency, etc.)
   - Organic intervention recommendations with priority levels

2. **📈 Yield Forecasting**
   - AI-projected harvest quantities with confidence intervals
   - Quality grading (Premium/Grade A/B/Standard)
   - Market valuation in INR based on current conditions

3. **💰 ROI Analysis**
   - Financial projections with setup costs & operating expenses
   - Business model recommendations (B2B, D2C, subscription, education)
   - Scaling strategies for urban expansion

4. **🎛️ What-If Simulator**
   - Interactive parameter adjustment (moisture, nutrients, light, temp)
   - Instant AI recalculation of health status & interventions
   - Scenario testing without re-training models

5. **🧠 Scenario Universe Generator**
   - Generates 3-12 variants per track:
     - Irrigation playbooks
     - Nutrient programs (Panchgavya/Jeevamrut cycles)
     - Lighting profiles
     - Resilience protocols
     - Commercial blueprints
     - Learning modules
   - Cross-linkage explanations
   - Combinatorial grid (up to 400 combinations displayed)

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Set your Groq API key:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-key-here"

# Linux/Mac
export GROQ_API_KEY="your-key-here"
```

Get a free API key from: https://console.groq.com/keys

### Running the Dashboard
```bash
streamlit run app.py --server.port 8503
```

### Training Models (Optional)
```bash
python train_model.py
```
Models auto-train on first dashboard load if missing.

## Project Structure
```
├── app.py                          # Main Streamlit dashboard
├── generate_data.py                # Improved data simulation
├── train_model.py                  # ML model training
├── groq_playbooks.py               # Groq API integration
├── requirements.txt                # Dependencies
├── PROJECT.md                      # Academic documentation
└── README.md                       # This file
```

## Presentation Highlights

### For Tomorrow's Demo

**1. Start with Problem Statement** (30 sec)
- Urban India: shrinking farmland + chemical dependency
- Show PROJECT.md slides (attach images in your deck)

**2. Data + Models** (1 min)
- Regenerate dataset → Train models → Show 99% irrigation accuracy
- Explain improved growth formula with temporal effects

**3. AI Diagnosis Live Demo** (2 min)
- Select crop (Tomato) + day (~30)
- Click "🔬 Run AI Diagnosis"
- Expand risk factors & interventions
- **WOW moment**: Show organic inputs (Panchgavya/neem/vermicompost)

**4. Yield + ROI Calculator** (2 min)
- Click "📊 Generate Yield Forecast"
- Show projected kg, market value in ₹, quality grade
- Adjust setup costs in ROI calculator
- Display business models (B2B office deliveries, rooftop kits, school programs)

**5. What-If Simulator** (1 min)
- Slide moisture to 30% (drought scenario)
- Click "🔮 Run What-If Analysis"
- Show status drops to "critical" with intervention priority changes
- **WOW moment**: Real-time AI recalculation without retraining

**6. Scenario Universe** (2 min)
- Set variants to 8-10
- Select all emphasis tracks
- Click "Generate AI scenario universe"
- Scroll through irrigation playbooks, nutrient programs, commercial blueprints
- Show combination grid (e.g., 6 irrigation × 5 nutrients × 4 lighting = 120 combos)
- **WOW moment**: "This is the design space explosion your teacher asked for!"

**7. Closing** (30 sec)
- Mention scalability: can add real IoT sensors later
- Entrepreneurial angle: AI-as-a-service subscriptions
- SDG alignment (SDG 2, 11, 12, 13)

### Key Talking Points
✅ "We replaced expensive IoT hardware with AI that learns from simulated sensors"  
✅ "Groq's structured outputs guarantee type-safe JSON—no parsing errors"  
✅ "Every recommendation ties back to Indian organic inputs like Panchgavya"  
✅ "The ROI calculator shows this is commercially viable for rooftop farms"  
✅ "What-if simulator lets farmers test scenarios before committing resources"

## Technical Notes

### Why R² Improved to 0.7+
- Changed growth formula from linear to quadratic distance penalties
- Added temporal dependency (cumulative growth boost over days)
- Stronger nutrient effect weighting (0.30 vs 0.20)

### Groq API Details
- Model: `openai/gpt-oss-120b` (supports structured outputs)
- Response schemas enforce JSON compliance
- Cached with Streamlit `@st.cache_data` (TTL 600-3600s)
- Fail-fast error handling with custom `GroqPlaybookError`

### Security
- API key via environment variable (never commit `.groq-env` with real keys)
- No secrets in code or git history

## Troubleshooting

**Q: "Missing GROQ_API_KEY" error**  
A: Set environment variable before running streamlit (see Quick Start)

**Q: Streamlit deprecation warnings**  
A: Already fixed—using `width="stretch"` instead of `use_container_width`

**Q: Models not loading**  
A: Click "📚 Train models now" in sidebar or run `python train_model.py`

**Q: Groq timeout/rate limit**  
A: Free tier has limits; reduce variant_cap or wait 60s between requests

## Future Enhancements
- [ ] Real sensor integration (ESP32 + MQTT)
- [ ] Multi-user authentication
- [ ] Historical trend analysis
- [ ] PDF report export
- [ ] Mobile app (React Native)

## License
Academic project for presentation purposes.

## Credits
- Groq for fast LLM inference
- Indian organic farming knowledge systems (Panchgavya, Jeevamrut)
- Streamlit for rapid prototyping
