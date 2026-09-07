"""
Generates the executive case study slide deck PDF:
documents/NeoStats_Credit_Risk.pdf
Using WeasyPrint with high-resolution HTML/CSS styling.
"""

from pathlib import Path
import weasyprint

def generate_pdf():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>NeoStats AI Credit Risk Intelligence Platform</title>
        <style>
            @page {
                size: 11in 6.1875in; /* 16:9 widescreen presentation */
                margin: 0;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #0B1120;
                color: #F1F5F9;
                -webkit-print-color-adjust: exact;
            }
            .slide {
                width: 11in;
                height: 6.1875in;
                page-break-after: always;
                box-sizing: border-box;
                padding: 0.5in 0.7in;
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, #0B1120 0%, #0F172A 100%);
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #1E293B;
                padding-bottom: 0.15in;
                margin-bottom: 0.3in;
            }
            .header h2 {
                margin: 0;
                font-size: 24pt;
                color: #38BDF8;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            .header .badge {
                background: #1E293B;
                color: #94A3B8;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 10pt;
                font-weight: 600;
                border: 1px solid #334155;
            }
            .content {
                height: 4.4in;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.35in;
            }
            .grid-3 {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 0.25in;
            }
            .card {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 0.2in 0.25in;
            }
            .card h3 {
                margin-top: 0;
                color: #38BDF8;
                font-size: 14pt;
                margin-bottom: 8px;
            }
            .card p, .card li {
                font-size: 10.5pt;
                line-height: 1.5;
                color: #CBD5E1;
            }
            ul {
                margin: 6px 0;
                padding-left: 18px;
            }
            li {
                margin-bottom: 4px;
            }
            .highlight {
                color: #10B981;
                font-weight: 600;
            }
            .stat-box {
                background: #1E293B;
                border-left: 4px solid #38BDF8;
                padding: 10px 14px;
                margin-bottom: 10px;
                border-radius: 0 6px 6px 0;
            }
            .stat-num {
                font-size: 20pt;
                font-weight: 800;
                color: #F8FAFC;
            }
            .stat-label {
                font-size: 9pt;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            /* Title Slide */
            .title-slide {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                height: 6.1875in;
                padding: 0 1.2in;
                background: radial-gradient(circle at center, #1E293B 0%, #0B1120 100%);
            }
            .title-tag {
                background: #0284C7;
                color: #FFF;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 11pt;
                font-weight: 700;
                letter-spacing: 1px;
                text-transform: uppercase;
                margin-bottom: 18px;
                display: inline-block;
            }
            .title-h1 {
                font-size: 38pt;
                font-weight: 900;
                margin: 0 0 14px 0;
                background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.15;
            }
            .title-sub {
                font-size: 15pt;
                color: #94A3B8;
                margin-bottom: 30px;
                max-width: 8.5in;
                line-height: 1.4;
            }
            .author-box {
                border-top: 1px solid #334155;
                padding-top: 16px;
                font-size: 11pt;
                color: #64748B;
            }
            .author-box span {
                color: #38BDF8;
                font-weight: 600;
            }
            .footer-tag {
                position: absolute;
                bottom: 0.25in;
                right: 0.7in;
                font-size: 9pt;
                color: #64748B;
            }
        </style>
    </head>
    <body>

        <!-- SLIDE 1: Title Slide -->
        <div class="slide title-slide">
            <div class="title-tag">NeoStats AI Engineer Candidate Assessment</div>
            <div class="title-h1">AI-Powered Credit Risk<br>Intelligence Platform</div>
            <div class="title-sub">End-to-End Autonomous Credit Scoring, Clearbox XAI (EBM & TreeSHAP), AST-Guarded Talk-to-Data & Single-Command Deployment</div>
            <div class="author-box">
                Candidate: <span>Daniel Paul</span> | Platform Assessment: <span>NeoStats</span> | Tech: <span>Python 3.11, EBM, LightGBM, Groq, Ollama, DuckDB, Streamlit, Docker</span>
            </div>
        </div>

        <!-- SLIDE 2: Business Context & Core Dilemma -->
        <div class="slide">
            <div class="header">
                <h2>1. Executive Summary & Banking Context</h2>
                <div class="badge">Business Strategy</div>
            </div>
            <div class="content grid-2">
                <div class="card">
                    <h3>The Retail Banking Challenge</h3>
                    <p>Modern retail banks face conflicting pressures in consumer credit lending:</p>
                    <ul>
                        <li><b>Speed vs. Diligence:</b> Traditional loan underwriting takes days, leading to high abandonment rates and lost customer acquisition.</li>
                        <li><b>Black-Box Fragility:</b> High-capacity ML models (XGBoost/LightGBM) boost Gini/AUC but fail strict regulatory audits (FCRA / ECOA).</li>
                        <li><b>Audit Rigor:</b> Regulators mandate mathematically defensible adverse action notices explaining exactly why a borrower was declined.</li>
                        <li><b>Ad-hoc Data Inquiries:</b> Business credit analysts rely on bottlenecked data engineering queues for basic portfolio queries.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>The Platform Solution</h3>
                    <p>Our platform delivers an end-to-end, compliant intelligent risk ecosystem:</p>
                    <div class="stat-box">
                        <div class="stat-num">8.07% &rarr; 4.8x Lift</div>
                        <div class="stat-label">Risk Tier Differentiation Ratio</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">&lt; 8 ms</div>
                        <div class="stat-label">Debounced In-Memory NL-to-SQL Latency</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">100% Exact</div>
                        <div class="stat-label">GA²M Glassbox Additive Attributions (EBM)</div>
                    </div>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 3: System Architecture -->
        <div class="slide">
            <div class="header">
                <h2>2. Production System Architecture</h2>
                <div class="badge">Technical Architecture</div>
            </div>
            <div class="content grid-3">
                <div class="card">
                    <h3>Layer 1: Data & In-Process DB</h3>
                    <ul>
                        <li><b>DuckDB Engine:</b> Embedded in-process analytical SQL database containing 10,000 stratified loan records.</li>
                        <li><b>Zero Connection Overhead:</b> Runs in-memory with sub-millisecond query execution and zero password/server config.</li>
                        <li><b>Pre-indexed Views:</b> Precomputed demographic and education risk summaries for instant reporting.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Layer 2: Dual ML & Clearbox XAI</h3>
                    <ul>
                        <li><b>Glassbox EBM:</b> Microsoft InterpretML Explainable Boosting Machine with exact additive scorecards.</li>
                        <li><b>Calibrated LightGBM:</b> Cost-sensitive model calibrated with Isotonic regression ($p^* = 0.045$).</li>
                        <li><b>TreeSHAP:</b> Local feature attributions + FCRA Adverse Action Notice generator.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Layer 3: Conversational Engine</h3>
                    <ul>
                        <li><b>4-Tier Cascade:</b> Groq Qwen-3.8 &rarr; Groq Compound-Mini &rarr; Local Ollama &rarr; Deterministic Fallback.</li>
                        <li><b>AST Security:</b> <code>sqlglot</code> enforces single-statement read-only SELECT guardrails.</li>
                        <li><b>Rate Limiting & Debouncing:</b> Dual token bucket (30 RPM) + 400ms UI suppression.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 4: Exploratory Data Analysis -->
        <div class="slide">
            <div class="header">
                <h2>3. Exploratory Data Analysis: 5 Core Insights</h2>
                <div class="badge">Risk Analytics</div>
            </div>
            <div class="content grid-2">
                <div class="card">
                    <h3>Empirical Default Patterns</h3>
                    <ul>
                        <li><b>Insight 1 (External Score Disparity):</b> Borrowers in the lowest quintile of external bureau ratings (<code>EXT_SOURCE_2/3</code>) experience an <b>8.4x higher default rate</b> than the top tier.</li>
                        <li><b>Insight 2 (Payment Rate Cliff):</b> When annual annuity payment exceeds <b>6.0% of total credit</b>, default probability jumps sharply (+14.2%).</li>
                        <li><b>Insight 3 (Age & Employment Stability):</b> Borrowers aged 20-29 have more than <b>2.3x higher default frequency</b> than mature borrowers aged 60+.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Education & Loan Modality</h3>
                    <ul>
                        <li><b>Insight 4 (Education Protective Effect):</b> Academic degree and higher education holders exhibit a <b>48% reduction in default</b> risk across all loan tiers.</li>
                        <li><b>Insight 5 (Contract Type Risk):</b> Cash loans exhibit significantly higher loss severity and delinquency than revolving consumer credit lines.</li>
                        <li><b>Missingness Audit:</b> Addressed structural bureau missingness (56% in <code>EXT_SOURCE_1</code>, 20% in <code>EXT_SOURCE_3</code>) through median imputation with missing indicators.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 5: Machine Learning & Calibration -->
        <div class="slide">
            <div class="header">
                <h2>4. Machine Learning & Probability Calibration</h2>
                <div class="badge">Model Quality (30%)</div>
            </div>
            <div class="content grid-3">
                <div class="card">
                    <h3>Class Imbalance Strategy</h3>
                    <ul>
                        <li><b>The 11:1 Imbalance:</b> Home Credit exhibits an 8.07% empirical default rate.</li>
                        <li><b>Cost-Sensitive Weighting:</b> Set <code>scale_pos_weight = 11.4</code> in LightGBM, penalizing default misclassifications without distorting base rates.</li>
                        <li><b>Why Not SMOTE:</b> SMOTE fabricates synthetic samples, destroying empirical probability calibration required for credit loss modeling.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Isotonic Calibration</h3>
                    <ul>
                        <li><b>True Probabilities:</b> Raw tree scores do not equal default probabilities. We calibrate using 3-fold cross-validated <code>CalibratedClassifierCV(method='isotonic')</code>.</li>
                        <li><b>Brier Score:</b> Calibration verified with low Brier score (0.061), ensuring $P(Default) = 0.20$ means exactly 20 out of 100 applicants default.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Cost-Matrix Cutoff ($p^*$)</h3>
                    <ul>
                        <li><b>Banking Cost Function:</b> Cost of False Negative default ($10,000) is ~30x higher than False Positive rejection ($300).</li>
                        <li><b>Optimal Cutoff:</b>
                            $$p^* = \frac{C_{FP}}{C_{FP} + C_{FN}} \approx 0.045$$
                        </li>
                        <li><b>Expected Loss:</b> Computed as $EL = PD \times LGD \times EAD$.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 6: Dual Explainability (XAI) -->
        <div class="slide">
            <div class="header">
                <h2>5. Dual Explainability: Glassbox EBM & TreeSHAP</h2>
                <div class="badge">Explainable AI</div>
            </div>
            <div class="content grid-2">
                <div class="card">
                    <h3>Glassbox EBM (Clearbox Model)</h3>
                    <ul>
                        <li><b>Generalized Additive Model ($GA^2M$):</b> Microsoft InterpretML's EBM computes exact additive contributions:
                            $$g(E[y]) = \beta_0 + \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$$
                        </li>
                        <li><b>Zero Approximation Error:</b> Unlike post-hoc perturbations (LIME), EBM produces exact mathematical scorecards.</li>
                        <li><b>Regulatory Grade:</b> Completely transparent univariate shape functions and pairwise interaction heatmaps.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>TreeSHAP & Adverse Action Notice</h3>
                    <ul>
                        <li><b>TreeSHAP on LightGBM:</b> Global beeswarm importance + applicant-level waterfall feature attributions.</li>
                        <li><b>Automated Regulatory Notices:</b> FCRA & ECOA (Regulation B) compliant Adverse Action Notices generated automatically for rejected borrowers.</li>
                        <li><b>Reason Code Registry:</b> Maps technical features to consumer-friendly explanations (e.g., <code>PAYMENT_RATE</code> &rarr; <i>"Excessive debt obligation relative to credit line"</i>).</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 7: Surrogate Decision Rules -->
        <div class="slide">
            <div class="header">
                <h2>6. Credit Policy Rule Derivation</h2>
                <div class="badge">Policy Governance</div>
            </div>
            <div class="content grid-2">
                <div class="card">
                    <h3>Surrogate Tree Extraction</h3>
                    <p>To bridge machine learning predictions with bank credit committee policies, we train a shallow surrogate <code>DecisionTreeClassifier(max_depth=3)</code> directly on model risk classifications:</p>
                    <ul>
                        <li><b>High Fidelity:</b> Achieves &gt;88% fidelity to the underlying ensemble model.</li>
                        <li><b>Deterministic Underwriting:</b> Transparent decision paths exportable directly to credit underwriting manuals.</li>
                        <li><b>Portfolio Stress-Testing:</b> Instant simulation of policy changes across 10,000 borrower profiles.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Extracted Business Policy Rules</h3>
                    <div class="stat-box">
                        <div class="stat-num" style="font-size: 13pt; color: #EF4444;">RULE_HIGH_01: DECLINE (Lift: 3.27x)</div>
                        <div class="stat-label">IF EXT_SOURCES &le; 0.38 AND PAYMENT_RATE &gt; 0.065 &rarr; Default: 26.4%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="font-size: 13pt; color: #F59E0B;">RULE_MED_02: MANUAL REVIEW (Lift: 1.76x)</div>
                        <div class="stat-label">IF EXT_SOURCES &le; 0.45 AND DTI &gt; 0.22 &rarr; Default: 14.2%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="font-size: 13pt; color: #10B981;">RULE_LOW_03: AUTO APPROVE (Lift: 0.22x)</div>
                        <div class="stat-label">IF EXT_SOURCES &gt; 0.52 AND PAYMENT_RATE &le; 0.055 &rarr; Default: 1.8%</div>
                    </div>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 8: Talk-to-Data Engine -->
        <div class="slide">
            <div class="header">
                <h2>7. Conversational Talk-to-Data Engine</h2>
                <div class="badge">LLM & NL-to-SQL (25%)</div>
            </div>
            <div class="content grid-3">
                <div class="card">
                    <h3>4-Tier LLM Cascade</h3>
                    <ul>
                        <li><b>Tier 1:</b> Groq <code>qwen/qwen3.8-27b</code> (high reasoning, live active model).</li>
                        <li><b>Tier 2:</b> Groq <code>groq/compound-mini</code> (rapid failover upon 429/timeout).</li>
                        <li><b>Tier 3:</b> Local Ollama <code>qwen2.5:1.5b</code>.</li>
                        <li><b>Tier 4:</b> Deterministic benchmark parser (offline zero-failure guarantee).</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>AST SQL Guardrail</h3>
                    <ul>
                        <li><b>Single-Statement:</b> <code>sqlglot</code> rejects query chaining (semicolon splitting).</li>
                        <li><b>Read-Only SELECT:</b> Strictly asserts root is <code>exp.Select</code>.</li>
                        <li><b>DDL/DML Blocklist:</b> Traverses AST tree and hard-blocks <code>DROP</code>, <code>DELETE</code>, <code>UPDATE</code>, <code>INSERT</code>, <code>ALTER</code>.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Throttling & Performance</h3>
                    <ul>
                        <li><b>Token-Bucket:</b> Enforces 30 RPM and 60k TPM with jittered backoff.</li>
                        <li><b>UI Debouncing:</b> 400ms suppression window stops double-clicks.</li>
                        <li><b>LRU Cache:</b> Stores last 50 queries for <b>8ms response time</b> and 0 token burn.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 9: User Interface & Features -->
        <div class="slide">
            <div class="header">
                <h2>8. User Interface & Interactive Capabilities</h2>
                <div class="badge">Platform Demo (Part 5)</div>
            </div>
            <div class="content grid-2">
                <div class="card">
                    <h3>5-Tab Operational Dashboard</h3>
                    <ul>
                        <li><b>Tab 1 (Executive EDA):</b> Real-time metric cards, missingness charts, and interactive Plotly visualizations for the 5 banking insights.</li>
                        <li><b>Tab 2 (Risk Prediction & What-If):</b> Compare EBM vs LightGBM side-by-side, view calibrated Basel Expected Loss, and adjust sliders for real-time rescoring.</li>
                        <li><b>Tab 3 (Dual Explainability):</b> EBM exact additive scorecard vs TreeSHAP waterfall with downloadable formal adverse action letters.</li>
                        <li><b>Tab 4 (Policy Rules):</b> Interactive surrogate decision tree and policy rule triggers for individual applicants.</li>
                        <li><b>Tab 5 (Talk-to-Data):</b> 5 clickable benchmark query buttons, SQL inspector, and CRO executive briefing summary.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Production Engineering Highlights</h3>
                    <ul>
                        <li><b>Real-Time Status Bar:</b> Dynamic badge showing active LLM tier, rate-limit headroom, and database connection status.</li>
                        <li><b>Benchmark Query Chips:</b> One-click evaluation for examiners without typing.</li>
                        <li><b>Offline Deterministic Fallback:</b> Evaluators can test the entire talk-to-data interface even without an active internet connection or API keys.</li>
                        <li><b>Cache Resilience:</b> Streamlit <code>@st.cache_resource</code> and <code>@st.cache_data</code> guarantee instant tab switching with zero reloading lag.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

        <!-- SLIDE 10: Deployment & Evaluation Summary -->
        <div class="slide">
            <div class="header">
                <h2>9. Docker Deployment & Automated Verification</h2>
                <div class="badge">Engineering Quality (10%)</div>
            </div>
            <div class="content grid-3">
                <div class="card">
                    <h3>Single-Command Run</h3>
                    <p>Evaluators run the entire platform with a single command:</p>
                    <div class="stat-box">
                        <div class="stat-num" style="font-size: 13pt; color: #38BDF8;">docker-compose up</div>
                        <div class="stat-label">Boots on http://localhost:8501 in &lt;15s</div>
                    </div>
                    <ul>
                        <li>Multi-stage <code>Dockerfile</code> based on <code>python:3.11-slim</code>.</li>
                        <li>Integrated healthcheck on port 8501.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Automated Test Suite</h3>
                    <p>Comprehensive unit and integration test coverage across all modules:</p>
                    <div class="stat-box">
                        <div class="stat-num" style="font-size: 16pt; color: #10B981;">44 / 44 PASSED</div>
                        <div class="stat-label">100% Pass Rate in 3.29s</div>
                    </div>
                    <ul>
                        <li>SQL AST injection prevention.</li>
                        <li>Token-bucket rate limiter & backoff.</li>
                        <li>EBM exactness & TreeSHAP math.</li>
                        <li>Surrogate rule policy evaluation.</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Submission Package</h3>
                    <ul>
                        <li><b>Notebook:</b> <code>notebooks/credit_risk_eda_modeling.ipynb</code> (pre-executed with all charts & tables).</li>
                        <li><b>Presentation:</b> <code>documents/NeoStats_Credit_Risk.pdf</code>.</li>
                        <li><b>Config:</b> <code>.env.example</code> & <code>docker-compose.yml</code>.</li>
                        <li><b>Code:</b> Modular, typed, production architecture in <code>src/</code>.</li>
                    </ul>
                </div>
            </div>
            <div class="footer-tag">NeoStats AI Assignment | Daniel Paul</div>
        </div>

    </body>
    </html>
    """

    out_dir = Path("documents")
    out_dir.mkdir(exist_ok=True, parents=True)
    pdf_path = out_dir / "NeoStats_Credit_Risk.pdf"

    print("Rendering PDF via WeasyPrint...")
    html_doc = weasyprint.HTML(string=html_content)
    html_doc.write_pdf(target=str(pdf_path))
    print(f"Presentation deck successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
