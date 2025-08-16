# 🍺 Fantasy Brewing Draft Advisor

A comprehensive draft tool for fantasy brewing competitions, featuring real-time draft tracking, style recommendations, opponent analytics, and trade management.

## 🎯 Overview

The Fantasy Brewing Draft Advisor helps participants in fantasy brewing leagues manage their ingredient drafts strategically. Players draft ingredients over multiple rounds to build competitive brewing teams, with the app providing intelligent recommendations, opponent analysis, and real-time draft management.

### Draft Rules
- **7 rounds** (+ optional 8th "Oh Sh*t" round for swaps)
- **Snake draft format** (reverse order each round)
- **Required ingredients**: 1 malt, 1 hop, 1 yeast, 1 adjunct + 3 flex picks
- **Trading allowed** with commissioner approval
- **Water, finings, rice hulls, water salts, acids, and yeast nutrients** are free (not drafted)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/willchase1/fantasybrewdraft.git
   cd fantasybrewdraft
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas
   ```

3. **Run the application**
   ```bash
   streamlit run fantasy_brewing_draft_appv2.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📊 Data Files

The app requires several data files to function properly:

### Required Files
- **`ingredients_2025.csv`** - Available ingredients by category
- **`style_matrix.json`** - Beer styles and their ingredient requirements
- **`ingredient_scarcity.json`** - Scarcity scoring for ingredients

### Optional Files
- **`opponent_model.json`** - Historical opponent picking patterns
- **`style_bias.json`** - Style preference weighting

### Auto-Generated Files
- **`draft_autosave.json`** - Current draft state (auto-saved)
- **`draft_state.json`** - Shared state for external tools

## 🎮 Features

### 🕐 Draft Timer
- **Adjustable duration** (1-60 minutes)
- **Visual countdown** with color-coded warnings
- **Start/Pause/Reset** controls
- **Elapsed time tracking**

### 🎯 Draft Management
- **Undo last pick** - Quick mistake correction
- **Swap/Trade manager** - Handle trades and Round 8 swaps
- **Auto-save** - Never lose your draft progress
- **Player tracking** - Monitor all participants

### 📈 Analytics & Recommendations

#### Draft Board
- **Live ingredient availability** by category
- **Popularity metrics** (picks, average draft slot)
- **Quick filtering** to find specific ingredients
- **Hover effects** for better UX

#### Style Viability
- **Real-time style scoring** based on your picks
- **Remaining options** for each beer style
- **Strategic guidance** for draft direction

#### Best Next Picks
- **Intelligent recommendations** using multiple factors:
  - Style coverage potential
  - Ingredient scarcity
  - Opponent picking tendencies
  - Category requirements
- **Opponent-aware** suggestions with bias weighting

#### Blocks & Opponent Predictions
- **Player tendency analysis** - Likely styles and next picks
- **Block suggestions** - Deny opponent builds
- **Historical pattern matching** using opponent model

#### Mock Draft Simulator
- **Full draft simulation** with customizable scenarios
- **Opponent AI** based on historical patterns
- **Different run scenarios** (base malt run, yeast run)
- **Style viability analysis** of simulated results

## ⚙️ Configuration

### Sidebar Controls

#### Draft Setup
- **Number of players** (4-20)
- **Player names** for each seat
- **Your draft position** (determines pick order)
- **Optional 8th round** toggle

#### Room Bias
- **Off/Conservative/Aggressive** - How strongly to weight opponent tendencies
- Requires `opponent_model.json` for full functionality

#### Category Requirements
- **Flex slots** (0-5) - Additional picks beyond required categories

#### Session Management
- **Reload data files** - Refresh CSV/JSON data
- **Reset session** - Clear all picks and start over

### Draft Status Panel
- **Pick tracking** - Used/remaining picks
- **Flex slots** - Available flexible picks
- **Required categories** - Visual indicators for draft requirements
- **Feasibility check** - Warns if requirements can't be met

## 🔄 Draft Management

### Undo Function
- **Single-click undo** of the most recent pick
- **Automatic state sync** across all components
- **Disabled when no picks** to prevent errors

### Swap/Trade Manager
1. **Open manager** - Click "🔄 Manage Swaps/Trades"
2. **Select pick** - Choose any existing pick to modify
3. **Choose replacement** - Select from available ingredients
4. **Execute swap** - Confirm the change

**Use cases:**
- Round 8 ingredient swaps
- Trade execution between players
- Commissioner pick corrections

## 📋 Tabs Overview

### 1. Draft Board
Main drafting interface with ingredient selection and draft tracking.

### 2. Style Viability
Real-time analysis of which beer styles remain viable with your current picks.

### 3. Recommendations
AI-powered suggestions for your next picks based on multiple strategic factors.

### 4. Blocks
Opponent analysis and suggestions for denying their builds.

### 5. Mock Draft Simulator
Full draft simulation to test different scenarios and strategies.

### 6. Results / Export
Team summaries, draft results, and export functionality (CSV/Excel).

## 🛠️ Customization

### Adding New Ingredients
1. Edit `ingredients_2025.csv` with new ingredients in appropriate columns
2. Update `style_matrix.json` if new beer styles are added
3. Refresh data using "Reload data files" button

### Custom Beer Styles
1. Edit `style_matrix.json` to add new styles or modify existing ones
2. Format: `"Style Name": {"Category": ["ingredient1", "ingredient2"]}`

### Opponent Modeling
Create `opponent_model.json` with historical data:
```json
{
  "ingredient_popularity": [
    {"Ingredient": "Citra", "Picks": 15, "Avg_Slot": 12.3, "Early_Score": 0.8}
  ],
  "top_pairs": [
    {"A": "Citra", "B": "Mosaic", "PairCount": 8}
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

**App won't start**
- Check Python version (3.8+ required)
- Install missing packages: `pip install streamlit pandas`

**Data not loading**
- Verify all required CSV/JSON files exist
- Check file formatting and encoding (UTF-8)
- Use "Reload data files" button after fixes

**Picks not saving**
- Check file write permissions in app directory
- Ensure `draft_autosave.json` can be created/modified

**Timer not updating**
- Refresh the browser page
- Check browser console for JavaScript errors

**Hover effects not working**
- Try refreshing the page
- Check if browser supports CSS transitions

### Performance Tips
- **Large player counts** (15+) may slow recommendations
- **Complex opponent models** increase calculation time
- **Browser refresh** if app becomes unresponsive

## 📁 File Structure

```
fantasybrewdraft/
├── fantasy_brewing_draft_appv2.py    # Main application
├── draft_state.py                    # State management utilities
├── Rules:.md.markdown                # Draft rules reference
├── ingredients_2025.csv              # Available ingredients
├── style_matrix.json                 # Beer style definitions
├── ingredient_scarcity.json          # Scarcity scoring
├── opponent_model.json               # Opponent patterns (optional)
├── style_bias.json                   # Style preferences (optional)
├── draft_autosave.json               # Auto-saved draft state
└── README.md                         # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [Pandas](https://pandas.pydata.org/) for data manipulation
- Inspired by fantasy sports draft tools and brewing competitions

---

**Need help?** Open an issue on GitHub or check the troubleshooting section above.