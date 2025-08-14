# VLM Analysis Aggregation Script

## 📊 **Current Aggregation Features**

### **Field Type Handling**
1. **Options Fields** (controlled vocabulary):
   - **Frequency counts**: `{"Male": 45, "Female": 23, "Non-binary": 2}`
   - **Percentages**: `{"Male": 64.3%, "Female": 32.9%, "Non-binary": 2.8%}`

2. **Range Fields** (numerical values):
   - **Statistical measures**: mean, median, std_dev, min, max
   - **Count** of valid numeric responses

3. **Open Fields** (free text):
   - **Word extraction**: Meaningful words excluding stop words
   - **Top words**: 20 most frequent words with counts
   - **Full frequency map**: All words with occurrence counts
   - **Stats**: total_words, unique_words

### **Schema-Aware Analysis**
- **People Array**: Special handling for multiple people per video
  - Total people across all videos
  - People per video distribution
  - Aggregated demographics, appearance, role_and_agency

- **Nested Objects**: Support for complex structures like `hair.length`, `hair.style`

- **Batch Processing**: Handles entire vlm_analysis directories or specific prompt groups

## 🚀 **Additional Aggregation Ideas**

### **1. Cross-Category Correlations**
```python
# Example correlations to analyze:
correlations = {
    "age_vs_clothing_style": "Do younger people wear more casual clothing?",
    "setting_vs_atmosphere": "Home settings → tender atmosphere?",
    "gender_vs_agency_level": "Agency level distribution by gender",
    "lighting_vs_mood": "Low key lighting → tense atmosphere?"
}
```

### **2. Cultural Pattern Detection**
```python
cultural_patterns = {
    "representation_bias": "Frequency of different racial/ethnic groups",
    "gender_role_patterns": "Agency levels by gender presentation",
    "objectification_indicators": "Male gaze levels vs sexualization levels",
    "stereotype_prevalence": "Common stereotype contexts by demographic"
}
```

### **3. Temporal/Sequential Analysis**
```python
sequential_analysis = {
    "video_progression": "How do metrics change across video numbers in a prompt?",
    "prompt_evolution": "How do different prompt groups compare?",
    "consistency_metrics": "How consistent are the same scenes across retries?"
}
```

### **4. Visual Composition Patterns**
```python
composition_insights = {
    "framing_by_genre": "Two Shot more common in Romance genre?",
    "lighting_mood_correlation": "Soft lighting → romantic atmosphere?",
    "color_emotion_mapping": "Warm colors → positive emotions?",
    "camera_movement_narrative": "Static shots for intimate scenes?"
}
```

### **5. Quality Assurance Metrics**
```python
qa_metrics = {
    "confidence_distribution": "Confidence scores across all fields",
    "undetectable_frequency": "Which fields are hardest to detect?",
    "response_completeness": "Percentage of fields with valid responses",
    "inter_video_consistency": "Similar videos → similar analyses?"
}
```

### **6. Comparative Analysis**
```python
comparative_features = {
    "prompt_group_comparison": "Compare metrics across different prompt groups",
    "demographic_representation": "Representation balance across categories",
    "cultural_flag_severity": "Distribution of bias levels",
    "narrative_agency_analysis": "Who has agency in different genres?"
}
```

## 📈 **Enhanced Output Structure**
```json
{
  "metadata": {...},
  "aggregated_data": {
    "people": {...},
    "composition": {...},
    "setting": {...},
    "cultural_flags": {...},
    "overall_notes": {...}
  },
  "insights": {
    "correlations": {...},
    "patterns": {...},
    "quality_metrics": {...},
    "comparative_analysis": {...}
  }
}
```

## 🎯 **Priority Enhancements**

1. **Correlation Analysis**: Age vs clothing, setting vs mood, lighting vs atmosphere
2. **Cultural Bias Detection**: Systematic analysis of representation patterns
3. **Quality Metrics**: Confidence distributions, "Undetectable" frequency analysis
4. **Prompt Group Comparison**: Statistical differences between prompt groups
5. **Export Formats**: CSV export for statistical software, visualization-ready JSON

## 🔧 **Implementation Questions**

1. **Correlation depth**: Should we calculate correlation coefficients or just cross-tabulations?
2. **Cultural analysis**: Any specific bias patterns you want to prioritize?
3. **Statistical exports**: Need CSV/Excel output for external analysis tools?
4. **Visualization integration**: Should the script generate plots/charts directly?
5. **Comparative analysis**: Compare against baseline datasets or just internal comparisons?

The current implementation provides a solid foundation for statistical analysis. Let me know which additional features would be most valuable for your research!
