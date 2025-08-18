import React from 'react';
import './WordCloudModal.css';

const WordCloudModal = ({ isOpen, onClose, data, title }) => {
  if (!isOpen) return null;

  // Extract word data for display
  const getWordData = () => {
    let wordData = null;

    if (data?.data?.top_words) {
      wordData = data.data.top_words;
    } else if (data?.data?.word_frequency) {
      wordData = data.data.word_frequency;
    } else if (data?.top_words) {
      wordData = data.top_words;
    } else if (data?.word_frequency) {
      wordData = data.word_frequency;
    }

    if (!wordData || typeof wordData !== 'object') {
      return [];
    }

    // Convert to sorted array for display
    return Object.entries(wordData)
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count); // Sort by count descending
  };

  const wordEntries = getWordData();

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="wordcloud-modal-backdrop" onClick={handleBackdropClick}>
      <div className="wordcloud-modal">
        <div className="wordcloud-modal-header">
          <h3>{title} - Word Details</h3>
          <button 
            className="wordcloud-modal-close" 
            onClick={onClose}
            aria-label="Close modal"
          >
            ×
          </button>
        </div>

        <div className="wordcloud-modal-content">
          {wordEntries.length > 0 ? (
            <div className="wordcloud-details">
              <div className="wordcloud-summary">
                <p><strong>Total unique words:</strong> {wordEntries.length}</p>
                <p><strong>Total occurrences:</strong> {wordEntries.reduce((sum, entry) => sum + entry.count, 0)}</p>
              </div>

              <div className="wordcloud-entries">
                {wordEntries.map((entry, idx) => (
                  <div key={idx} className="wordcloud-entry">
                    <div className="wordcloud-entry-word">{entry.word}</div>
                    <div className="wordcloud-entry-count">{entry.count}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="wordcloud-no-data">
              No word data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WordCloudModal;
