document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('inductionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.querySelector('.btn-text');
    const btnIcon = document.querySelector('.btn-icon');
    const btnLoader = document.getElementById('btnLoader');
    
    const emptyState = document.getElementById('emptyState');
    const resultSection = document.getElementById('resultSection');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Result DOM elements
    const predictedBadge = document.getElementById('predictedBadge');
    const resPredicate = document.getElementById('resPredicate');
    const resLemma = document.getElementById('resLemma');
    const argumentsList = document.getElementById('argumentsList');
    const argsEmpty = document.getElementById('argsEmpty');
    const similarityTableBody = document.getElementById('similarityTableBody');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const sentence = document.getElementById('sentence').value.trim();
        const target_predicate = document.getElementById('predicate').value.trim();
        
        if (!sentence || !target_predicate) return;

        // Reset state
        hideError();
        setLoadingState(true);
        
        // Hide empty state & results gracefully
        emptyState.classList.add('hidden');
        resultSection.classList.add('hidden');
        
        // Brief timeout ensures CSS transitions catch the DOM update
        setTimeout(async () => {
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sentence: sentence,
                        target_predicate: target_predicate,
                        embed_type: "minilm", 
                        num_similar: 5
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction failed.');
                }

                renderResults(data);
                
            } catch (error) {
                showError(error.message);
                // Return to empty state if this is the first run
                if(predictedBadge.textContent === '--') {
                    emptyState.classList.remove('hidden');
                }
            } finally {
                setLoadingState(false);
            }
        }, 300);
    });

    function setLoadingState(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = 'Processing...';
            btnIcon.classList.add('hidden');
            btnLoader.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            btnText.textContent = 'Execute Induction';
            btnIcon.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    }

    function renderResults(data) {
        // Trigger reflow for animations
        resultSection.classList.remove('hidden');
        
        // Animate elements individually
        const widgets = document.querySelectorAll('.widget');
        widgets.forEach(w => {
            w.style.animation = 'none';
            w.offsetHeight; // trigger reflow
            w.style.animation = null; 
        });

        // 1. Predicted Frame
        predictedBadge.textContent = data.predicted_frame;

        // 2. Morphology
        resPredicate.textContent = data.predicate_info.text;
        resLemma.textContent = data.predicate_info.lemma;

        // 3. Arguments
        argumentsList.innerHTML = '';
        if (data.arguments && data.arguments.length > 0) {
            argsEmpty.classList.add('hidden');
            data.arguments.forEach(arg => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="arg-text">${arg.text}</span>
                    <span class="arg-role">${arg.role}</span>
                `;
                argumentsList.appendChild(li);
            });
        } else {
            argsEmpty.classList.remove('hidden');
        }

        // 4. Similarity Table
        similarityTableBody.innerHTML = '';
        data.similar_examples.forEach((ex, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="sim-val">${ex.similarity}</span></td>
                <td>${ex.sentence}</td>
                <td><span class="gold-chip">${ex.gold_frame}</span></td>
            `;
            similarityTableBody.appendChild(tr);
        });
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
        errorText.textContent = '';
    }
});
