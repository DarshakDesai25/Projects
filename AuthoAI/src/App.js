import React, { useState, useMemo } from 'react';
import patientDataset from './patient_dataset.json';

// ============================================================================
// AuthoAI — Massachusetts Standard Prior Authorization Auto-Filler
// Single-file React component, inline styles only.
// ============================================================================

const BLUE = '#1a5276';
const BLUE_LIGHT = '#d4e6f1';
const BORDER = '#b0bec5';
const BG = '#f4f6f8';

function App() {
  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------
  const [memberId, setMemberId] = useState('');
  const [patientData, setPatientData] = useState(null);
  const [lookupMessage, setLookupMessage] = useState('');

  const [physicianName, setPhysicianName] = useState('');
  const [primaryDiagnosis, setPrimaryDiagnosis] = useState('');
  const [requestedMedication, setRequestedMedication] = useState('');
  const [drugAllergies, setDrugAllergies] = useState('');
  const [additionalNotes, setAdditionalNotes] = useState('');

  const [llmData, setLlmData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [copyStatus, setCopyStatus] = useState('');

  // --------------------------------------------------------------------------
  // Derived values
  // --------------------------------------------------------------------------
  const missingFields = useMemo(() => {
    const missing = [];
    if (!patientData) missing.push('Member ID (patient not loaded)');
    if (!physicianName.trim()) missing.push('Requesting Physician Name');
    if (!primaryDiagnosis.trim()) missing.push('Primary Diagnosis');
    if (!requestedMedication.trim()) missing.push('Requested Medication');
    if (!drugAllergies.trim()) missing.push('Drug Allergies');
    if (!additionalNotes.trim()) missing.push('Additional Notes');
    return missing;
  }, [patientData, physicianName, primaryDiagnosis, requestedMedication, drugAllergies, additionalNotes]);

  const approvalScore = useMemo(() => {
    let score = 40;
    if (patientData) score += 8;
    if (physicianName.trim()) score += 5;
    if (primaryDiagnosis.trim()) score += 7;
    if (requestedMedication.trim()) score += 7;
    if (drugAllergies.trim()) score += 4;
    if (additionalNotes.trim()) score += 4;
    if (llmData) {
      if (llmData.icd10_code) score += 5;
      if (llmData.dosing_schedule) score += 3;
      if (llmData.length_of_therapy) score += 2;
      if (llmData.clinical_justification) score += 6;
      if (llmData.medical_necessity_statement) score += 5;
      if (llmData.previous_therapy_failure_description) score += 2;
    }
    return Math.min(score, 98);
  }, [patientData, physicianName, primaryDiagnosis, requestedMedication, drugAllergies, additionalNotes, llmData]);

  const scoreColor =
    approvalScore >= 80 ? '#2e7d32' : approvalScore >= 60 ? '#f9a825' : '#c62828';

  // --------------------------------------------------------------------------
  // Member ID lookup
  // --------------------------------------------------------------------------
  const normalizeMemberId = (value) =>
    value.trim().replace(/[-\s]/g, '').toUpperCase();

  const handleLookup = () => {
    setErrorMessage('');
    setLlmData(null);
    const normalizedId = normalizeMemberId(memberId);
    if (!normalizedId) {
      setPatientData(null);
      setLookupMessage('');
      return;
    }

    const exactMatch = patientDataset[normalizedId];
    if (exactMatch) {
      setPatientData(exactMatch);
      setLookupMessage(`✓ Patient found: ${exactMatch.patient_name}`);
      return;
    }

    const digitsOnly = normalizedId.replace(/\D/g, '');
    const digitMatch = Object.values(patientDataset).find(
      (patient) => patient.member_id.replace(/\D/g, '') === digitsOnly
    );

    if (digitMatch) {
      setPatientData(digitMatch);
      setLookupMessage(`✓ Patient found: ${digitMatch.patient_name} (matched by numeric member ID)`);
      return;
    }

    setPatientData(null);
    setLookupMessage(
      `✗ No patient found for Member ID "${memberId.trim()}". ` +
      'Try using a valid ID like MA630297375.'
    );
  };

  // --------------------------------------------------------------------------
  // Ollama generate
  // --------------------------------------------------------------------------
  const buildPrompt = () => {
    const prev = (patientData.previous_therapies_tried || [])
      .map((t, i) => `  ${i + 1}. ${t.drug} ${t.strength} — ${t.schedule} — Failed: ${t.reason}`)
      .join('\n');
    const concurrent = (patientData.pertinent_concurrent_medications || []).join(', ');

    return `You are a clinical documentation assistant helping fill a Massachusetts Standard Prior Authorization form.

PATIENT INFORMATION:
- Name: ${patientData.patient_name}
- DOB: ${patientData.date_of_birth}
- Gender: ${patientData.gender}
- Concurrent Medications: ${concurrent || 'None listed'}

PREVIOUS THERAPIES TRIED:
${prev || '  None recorded'}

PHYSICIAN-PROVIDED INFORMATION:
- Requesting Physician: ${physicianName}
- Primary Diagnosis: ${primaryDiagnosis}
- Requested Medication: ${requestedMedication}
- Drug Allergies: ${drugAllergies}
- Additional Clinical Notes: ${additionalNotes}

TASK:
Based on the information above, infer clinically appropriate values for the following fields and return a JSON object. Use standard medical reasoning. For ICD-10, pick the most specific code matching the primary diagnosis. For dosing, use standard FDA-approved dosing for the requested medication. Write clinical_justification and medical_necessity_statement as professional prose (2-4 sentences each) citing the patient's history, failed therapies, and clinical rationale.

Return ONLY a valid JSON object with no explanation, no markdown, no backticks. Use exactly these keys:

{
  "icd10_code": "",
  "dosing_schedule": "",
  "length_of_therapy": "",
  "quantity": "",
  "date_therapy_initiated": "",
  "is_compound": "",
  "off_label_use": "",
  "peer_reviewed_citation": "",
  "pertinent_comorbidities": "",
  "contraindications": "",
  "nonpharmacologic_therapies": "",
  "lab_values": "",
  "clinical_justification": "",
  "medical_necessity_statement": "",
  "previous_therapy_failure_description": "",
  "opioid_management_tools": ""
}`;
  };

  const safeParseJson = (text) => {
    if (!text) return null;
    // Strip markdown fences if model ignored the instruction
    let cleaned = text.trim();
    cleaned = cleaned.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
    // Extract first { ... } block to be robust against preamble
    const first = cleaned.indexOf('{');
    const last = cleaned.lastIndexOf('}');
    if (first === -1 || last === -1 || last < first) return null;
    const candidate = cleaned.slice(first, last + 1);
    try {
      return JSON.parse(candidate);
    } catch (e) {
      return null;
    }
  };

  const handleGenerate = async () => {
    setErrorMessage('');
    setCopyStatus('');
    if (missingFields.length > 0) {
      setErrorMessage('Please fill all required fields before generating.');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'adrienbrault/biomistral-7b:Q5_K_M',
          prompt: buildPrompt(),
          stream: false,
          format: 'json',
          options: { temperature: 0.1 },
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama returned HTTP ${response.status}`);
      }

      const data = await response.json();
      const parsed = safeParseJson(data.response);
      if (!parsed) {
        console.error('Raw Ollama response:', data.response);
        throw new Error('Ollama response could not be parsed as JSON. Check browser console for raw output.');
      }
      setLlmData(parsed);
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        setErrorMessage('Cannot reach Ollama at http://localhost:11434. Please run: ollama serve && ollama pull mistral');
      } else {
        setErrorMessage(`Generation failed: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------------------------------
  // Copy letter (medical necessity + justification)
  // --------------------------------------------------------------------------
  const handleCopyLetter = async () => {
    if (!llmData || !patientData) return;
    const letter = buildLetterText(patientData, {
      physicianName,
      primaryDiagnosis,
      requestedMedication,
      drugAllergies,
      additionalNotes,
    }, llmData);
    try {
      await navigator.clipboard.writeText(letter);
      setCopyStatus('✓ Letter copied to clipboard');
      setTimeout(() => setCopyStatus(''), 2500);
    } catch (e) {
      setCopyStatus('✗ Copy failed');
    }
  };

  // --------------------------------------------------------------------------
  // Download PDF (via print in new tab)
  // --------------------------------------------------------------------------
  const handleDownloadPdf = () => {
    if (!patientData) return;
    const html = buildPrintHtml(patientData, {
      physicianName,
      primaryDiagnosis,
      requestedMedication,
      drugAllergies,
      additionalNotes,
    }, llmData || {});
    const win = window.open('', '_blank');
    if (!win) {
      setErrorMessage('Popup was blocked. Please allow popups to download the PDF.');
      return;
    }
    win.document.open();
    win.document.write(html);
    win.document.close();
    // Give the new window a moment to render before printing
    setTimeout(() => {
      try { win.focus(); win.print(); } catch (e) { /* noop */ }
    }, 400);
  };

  // --------------------------------------------------------------------------
  // Render
  // --------------------------------------------------------------------------
  return (
    <div style={styles.page}>
      {/* Header */}
      <header style={styles.header}>
        <div>
          <div style={styles.brand}>AuthoAI</div>
          <div style={styles.tagline}>
            Massachusetts Standard Prior Authorization — AI-Assisted Auto-Fill
          </div>
        </div>
        <div style={styles.headerRight}>
          <div style={{ ...styles.scoreBadge, backgroundColor: scoreColor }}>
            <div style={styles.scoreLabel}>Approval Likelihood</div>
            <div style={styles.scoreValue}>{approvalScore}%</div>
          </div>
        </div>
      </header>

      {/* Body: two panels */}
      <div style={styles.body}>
        {/* LEFT PANEL — input form */}
        <section style={styles.leftPanel}>
          <h2 style={styles.panelTitle}>Provider Input</h2>

          {/* Member ID Lookup */}
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Member ID</label>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                style={{ ...styles.input, flex: 1 }}
                placeholder="e.g. MA498444591"
                value={memberId}
                onChange={(e) => setMemberId(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleLookup(); }}
              />
              <button style={styles.secondaryBtn} onClick={handleLookup}>
                Look Up
              </button>
            </div>
            {lookupMessage && (
              <div style={{
                ...styles.lookupMsg,
                color: patientData ? '#2e7d32' : '#c62828',
              }}>
                {lookupMessage}
              </div>
            )}
          </div>

          {/* Auto-populated patient summary */}
          {patientData && (
            <div style={styles.patientCard}>
              <div style={styles.patientCardTitle}>Patient Record Loaded</div>
              <div style={styles.patientGrid}>
                <div><strong>Name:</strong> {patientData.patient_name}</div>
                <div><strong>DOB:</strong> {patientData.date_of_birth}</div>
                <div><strong>Gender:</strong> {patientData.gender}</div>
                <div><strong>Member ID:</strong> {patientData.member_id}</div>
                <div><strong>Plan:</strong> {patientData.health_plan?.name}</div>
                <div><strong>Specialty:</strong> {patientData.physician?.specialty}</div>
              </div>
              <div style={styles.patientSub}>
                <strong>Concurrent Meds:</strong>{' '}
                {(patientData.pertinent_concurrent_medications || []).join(', ') || '—'}
              </div>
              <div style={styles.patientSub}>
                <strong>Previous Therapies:</strong>{' '}
                {(patientData.previous_therapies_tried || []).map(t => t.drug).join(', ') || '—'}
              </div>
            </div>
          )}

          {/* 5 Doctor inputs */}
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Requesting Physician Name</label>
            <input
              style={styles.input}
              placeholder="Dr. Jane Smith, MD"
              value={physicianName}
              onChange={(e) => setPhysicianName(e.target.value)}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Primary Diagnosis</label>
            <input
              style={styles.input}
              placeholder="e.g. Type 2 Diabetes Mellitus, uncontrolled"
              value={primaryDiagnosis}
              onChange={(e) => setPrimaryDiagnosis(e.target.value)}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Requested Medication</label>
            <input
              style={styles.input}
              placeholder="e.g. Ozempic 1mg weekly"
              value={requestedMedication}
              onChange={(e) => setRequestedMedication(e.target.value)}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Drug Allergies</label>
            <input
              style={styles.input}
              placeholder="e.g. Penicillin (rash), Sulfa drugs — or 'NKDA'"
              value={drugAllergies}
              onChange={(e) => setDrugAllergies(e.target.value)}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Additional Notes</label>
            <textarea
              style={{ ...styles.input, minHeight: 80, fontFamily: 'inherit' }}
              placeholder="Any additional clinical context, labs, or observations…"
              value={additionalNotes}
              onChange={(e) => setAdditionalNotes(e.target.value)}
            />
          </div>

          {/* Missing fields warning */}
          {missingFields.length > 0 && (
            <div style={styles.warning}>
              <strong>Missing required fields:</strong>
              <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
                {missingFields.map((f) => <li key={f}>{f}</li>)}
              </ul>
            </div>
          )}

          {/* Error */}
          {errorMessage && (
            <div style={styles.error}>{errorMessage}</div>
          )}

          {/* Action buttons */}
          <div style={{ display: 'flex', gap: 8, marginTop: 16, flexWrap: 'wrap' }}>
            <button
              style={{
                ...styles.primaryBtn,
                opacity: loading || missingFields.length > 0 ? 0.6 : 1,
                cursor: loading || missingFields.length > 0 ? 'not-allowed' : 'pointer',
              }}
              onClick={handleGenerate}
              disabled={loading || missingFields.length > 0}
            >
              {loading ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                  <Spinner /> Generating with Mistral…
                </span>
              ) : 'Generate Clinical Fields'}
            </button>

            <button
              style={{
                ...styles.secondaryBtn,
                opacity: llmData ? 1 : 0.5,
                cursor: llmData ? 'pointer' : 'not-allowed',
              }}
              onClick={handleCopyLetter}
              disabled={!llmData}
            >
              Copy Letter
            </button>

            <button
              style={{
                ...styles.secondaryBtn,
                opacity: patientData ? 1 : 0.5,
                cursor: patientData ? 'pointer' : 'not-allowed',
              }}
              onClick={handleDownloadPdf}
              disabled={!patientData}
            >
              Download PDF
            </button>
          </div>
          {copyStatus && <div style={{ marginTop: 8, color: '#2e7d32' }}>{copyStatus}</div>}
        </section>

        {/* RIGHT PANEL — live preview of MA form */}
        <section style={styles.rightPanel}>
          <h2 style={styles.panelTitle}>Form Preview — MA Standard Prior Auth</h2>
          <MaFormPreview
            patient={patientData}
            inputs={{ physicianName, primaryDiagnosis, requestedMedication, drugAllergies, additionalNotes }}
            llm={llmData || {}}
          />
        </section>
      </div>

      <footer style={styles.footer}>
        AuthoAI runs fully offline · Ollama + Mistral · Massachusetts Standard Form for Medication Prior Authorization Requests
      </footer>
    </div>
  );
}

// ============================================================================
// MA Form Preview — Sections A through F
// ============================================================================
function MaFormPreview({ patient, inputs, llm }) {
  const blank = '—';
  const g = (v) => (v && String(v).trim() !== '' ? v : blank);

  return (
    <div style={styles.formPaper}>
      <div style={styles.formHeader}>
        <div style={{ fontSize: 16, fontWeight: 700 }}>
          Massachusetts Standard Form for Medication Prior Authorization Requests
        </div>
        <div style={{ fontSize: 11, color: '#555', marginTop: 4 }}>
          Submission of this form does not guarantee coverage. Response typically within 2 business days.
        </div>
      </div>

      {/* Section A */}
      <FormSection letter="A" title="Destination (Health Plan)">
        <Row label="Health Plan Name" value={g(patient?.health_plan?.name)} />
        <Row label="Phone" value={g(patient?.health_plan?.phone)} />
        <Row label="Fax" value={g(patient?.health_plan?.fax)} />
      </FormSection>

      {/* Section B */}
      <FormSection letter="B" title="Patient Information">
        <Row label="Patient Name" value={g(patient?.patient_name)} />
        <Row label="Date of Birth" value={g(patient?.date_of_birth)} />
        <Row label="Gender" value={g(patient?.gender)} />
        <Row label="Member ID" value={g(patient?.member_id)} />
      </FormSection>

      {/* Section C */}
      <FormSection letter="C" title="Prescriber Information">
        <Row label="Physician Name" value={g(inputs.physicianName)} />
        <Row label="Specialty" value={g(patient?.physician?.specialty)} />
        <Row label="Phone" value={g(patient?.physician?.phone)} />
        <Row label="NPI" value={g(patient?.physician?.npi_number)} />
        <Row label="DEA" value={g(patient?.physician?.dea_number)} />
      </FormSection>

      {/* Section D */}
      <FormSection letter="D" title="Medication Information">
        <Row label="Medication" value={g(inputs.requestedMedication)} />
        <Row label="Strength" value={g(extractStrength(inputs.requestedMedication))} />
        <Row label="Quantity" value={g(llm.quantity)} />
        <Row label="Dosing Schedule" value={g(llm.dosing_schedule)} />
        <Row label="Length of Therapy" value={g(llm.length_of_therapy)} />
        <Row label="Date Therapy Initiated" value={g(llm.date_therapy_initiated)} />
      </FormSection>

      {/* Section E */}
      <FormSection letter="E" title="Compound / Off-Label Use">
        <Row label="Is Compound" value={g(llm.is_compound)} />
        <Row label="Off-Label Use" value={g(llm.off_label_use)} />
        <Row label="Peer-Reviewed Citation" value={g(llm.peer_reviewed_citation)} />
      </FormSection>

      {/* Section F */}
      <FormSection letter="F" title="Clinical Information">
        <Row label="Primary Diagnosis" value={g(inputs.primaryDiagnosis)} />
        <Row label="ICD-10 Code" value={g(llm.icd10_code)} />
        <Row label="Pertinent Comorbidities" value={g(llm.pertinent_comorbidities)} />
        <Row label="Drug Allergies" value={g(inputs.drugAllergies)} />
        <Row
          label="Concurrent Medications"
          value={g((patient?.pertinent_concurrent_medications || []).join(', '))}
        />

        <div style={{ marginTop: 10 }}>
          <div style={styles.subLabel}>Previous Therapies Tried</div>
          <table style={styles.therapyTable}>
            <thead>
              <tr>
                <th style={styles.th}>Drug Name</th>
                <th style={styles.th}>Strength</th>
                <th style={styles.th}>Schedule</th>
                <th style={styles.th}>Reason for Failure</th>
              </tr>
            </thead>
            <tbody>
              {(patient?.previous_therapies_tried || []).length === 0 ? (
                <tr><td style={styles.td} colSpan={4}>—</td></tr>
              ) : (
                patient.previous_therapies_tried.map((t, i) => (
                  <tr key={i}>
                    <td style={styles.td}>{g(t.drug)}</td>
                    <td style={styles.td}>{g(t.strength)}</td>
                    <td style={styles.td}>{g(t.schedule)}</td>
                    <td style={styles.td}>{g(t.reason)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <Row label="Previous Therapy Failure Summary" value={g(llm.previous_therapy_failure_description)} />
        <Row label="Contraindications" value={g(llm.contraindications)} />
        <Row label="Nonpharmacologic Therapies" value={g(llm.nonpharmacologic_therapies)} />
        <Row label="Lab Values" value={g(llm.lab_values)} />
        <Row label="Opioid Management Tools" value={g(llm.opioid_management_tools)} />
        <Row label="Clinical Justification" value={g(llm.clinical_justification)} wide />
        <Row label="Medical Necessity Statement" value={g(llm.medical_necessity_statement)} wide />
        <Row label="Additional Notes" value={g(inputs.additionalNotes)} wide />
      </FormSection>
    </div>
  );
}

function FormSection({ letter, title, children }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionHeader}>
        <span style={styles.sectionLetter}>{letter}</span>
        <span>{title}</span>
      </div>
      <div style={styles.sectionBody}>{children}</div>
    </div>
  );
}

function Row({ label, value, wide }) {
  return (
    <div style={{ ...styles.row, flexDirection: wide ? 'column' : 'row' }}>
      <div style={{ ...styles.rowLabel, width: wide ? '100%' : 200 }}>{label}</div>
      <div style={{ ...styles.rowValue, whiteSpace: 'pre-wrap' }}>{value}</div>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================
function extractStrength(med) {
  if (!med) return '';
  const m = med.match(/\d+(?:\.\d+)?\s?(mg|mcg|g|ml|units?)/i);
  return m ? m[0] : '';
}

function Spinner() {
  return (
    <span
      style={{
        width: 14, height: 14,
        border: '2px solid #fff',
        borderTopColor: 'transparent',
        borderRadius: '50%',
        display: 'inline-block',
        animation: 'authoai-spin 0.8s linear infinite',
      }}
    />
  );
}

function buildLetterText(patient, inputs, llm) {
  return `PRIOR AUTHORIZATION REQUEST — LETTER OF MEDICAL NECESSITY

To: ${patient.health_plan?.name || ''}
Fax: ${patient.health_plan?.fax || ''}

Patient: ${patient.patient_name} | DOB: ${patient.date_of_birth} | Member ID: ${patient.member_id}
Prescriber: ${inputs.physicianName} | NPI: ${patient.physician?.npi_number || ''}
Requested Medication: ${inputs.requestedMedication}
Primary Diagnosis: ${inputs.primaryDiagnosis} (ICD-10: ${llm.icd10_code || 'pending'})

CLINICAL JUSTIFICATION:
${llm.clinical_justification || ''}

MEDICAL NECESSITY:
${llm.medical_necessity_statement || ''}

PREVIOUS THERAPY FAILURES:
${llm.previous_therapy_failure_description || ''}

Drug Allergies: ${inputs.drugAllergies}
Concurrent Medications: ${(patient.pertinent_concurrent_medications || []).join(', ')}

Additional Notes: ${inputs.additionalNotes}

Respectfully,
${inputs.physicianName}
`;
}

function buildPrintHtml(patient, inputs, llm) {
  const safe = (v) => (v && String(v).trim() !== '' ? String(v) : '—');
  const rows = (patient.previous_therapies_tried || []).map(t => `
    <tr>
      <td>${safe(t.drug)}</td><td>${safe(t.strength)}</td>
      <td>${safe(t.schedule)}</td><td>${safe(t.reason)}</td>
    </tr>`).join('') || `<tr><td colspan="4">—</td></tr>`;

  const section = (letter, title, inner) => `
    <div class="section">
      <div class="sec-head"><span class="sec-letter">${letter}</span>${title}</div>
      <div class="sec-body">${inner}</div>
    </div>`;

  const row = (label, value, wide = false) => `
    <div class="row ${wide ? 'wide' : ''}">
      <div class="lbl">${label}</div>
      <div class="val">${safe(value)}</div>
    </div>`;

  return `<!doctype html><html><head><meta charset="utf-8"/>
  <title>MA Prior Authorization — ${safe(patient.patient_name)}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Arial, Helvetica, sans-serif; color:#222; margin:24px; font-size:12px; }
    h1 { font-size:16px; margin:0 0 4px 0; }
    .sub { color:#555; font-size:11px; margin-bottom:14px; }
    .section { border:1px solid #b0bec5; margin-bottom:10px; break-inside: avoid; }
    .sec-head { background:${BLUE}; color:#fff; padding:6px 10px; font-weight:700; font-size:12px; }
    .sec-letter { display:inline-block; width:18px; height:18px; line-height:18px; text-align:center;
      background:#fff; color:${BLUE}; border-radius:3px; margin-right:8px; font-weight:800; }
    .sec-body { padding:8px 10px; }
    .row { display:flex; padding:3px 0; border-bottom:1px dotted #ddd; }
    .row.wide { flex-direction:column; }
    .row:last-child { border-bottom:none; }
    .lbl { width:200px; font-weight:700; color:#333; }
    .row.wide .lbl { width:100%; margin-bottom:2px; }
    .val { flex:1; white-space:pre-wrap; }
    table { width:100%; border-collapse: collapse; margin:6px 0; font-size:11px; }
    th { background:${BLUE_LIGHT}; color:${BLUE}; padding:5px; text-align:left; border:1px solid #b0bec5; }
    td { padding:5px; border:1px solid #b0bec5; }
    @media print { body { margin: 12mm; } }
  </style></head><body>
  <h1>Massachusetts Standard Form for Medication Prior Authorization Requests</h1>
  <div class="sub">Submission does not guarantee coverage. Response typically within 2 business days.</div>

  ${section('A', 'Destination (Health Plan)',
    row('Health Plan Name', patient.health_plan?.name) +
    row('Phone', patient.health_plan?.phone) +
    row('Fax', patient.health_plan?.fax))}

  ${section('B', 'Patient Information',
      row('Patient Name', patient.patient_name) +
      row('Date of Birth', patient.date_of_birth) +
      row('Gender', patient.gender) +
      row('Member ID', patient.member_id))}

  ${section('C', 'Prescriber Information',
        row('Physician Name', inputs.physicianName) +
        row('Specialty', patient.physician?.specialty) +
        row('Phone', patient.physician?.phone) +
        row('NPI', patient.physician?.npi_number) +
        row('DEA', patient.physician?.dea_number))}

  ${section('D', 'Medication Information',
          row('Medication', inputs.requestedMedication) +
          row('Strength', extractStrength(inputs.requestedMedication)) +
          row('Quantity', llm.quantity) +
          row('Dosing Schedule', llm.dosing_schedule) +
          row('Length of Therapy', llm.length_of_therapy) +
          row('Date Therapy Initiated', llm.date_therapy_initiated))}

  ${section('E', 'Compound / Off-Label Use',
            row('Is Compound', llm.is_compound) +
            row('Off-Label Use', llm.off_label_use) +
            row('Peer-Reviewed Citation', llm.peer_reviewed_citation))}

  ${section('F', 'Clinical Information',
              row('Primary Diagnosis', inputs.primaryDiagnosis) +
              row('ICD-10 Code', llm.icd10_code) +
              row('Pertinent Comorbidities', llm.pertinent_comorbidities) +
              row('Drug Allergies', inputs.drugAllergies) +
              row('Concurrent Medications', (patient.pertinent_concurrent_medications || []).join(', ')) +
              `<div style="margin-top:6px;"><div style="font-weight:700;color:#333;">Previous Therapies Tried</div>
      <table><thead><tr><th>Drug</th><th>Strength</th><th>Schedule</th><th>Reason for Failure</th></tr></thead>
      <tbody>${rows}</tbody></table></div>` +
              row('Previous Therapy Failure Summary', llm.previous_therapy_failure_description, true) +
              row('Contraindications', llm.contraindications, true) +
              row('Nonpharmacologic Therapies', llm.nonpharmacologic_therapies, true) +
              row('Lab Values', llm.lab_values, true) +
              row('Opioid Management Tools', llm.opioid_management_tools, true) +
              row('Clinical Justification', llm.clinical_justification, true) +
              row('Medical Necessity Statement', llm.medical_necessity_statement, true) +
              row('Additional Notes', inputs.additionalNotes, true))}

  </body></html>`;
}

// ============================================================================
// Styles
// ============================================================================
const styles = {
  page: {
    minHeight: '100vh',
    background: BG,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#222',
  },
  header: {
    background: BLUE,
    color: '#fff',
    padding: '16px 24px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  brand: { fontSize: 22, fontWeight: 800, letterSpacing: 0.5 },
  tagline: { fontSize: 12, opacity: 0.9, marginTop: 2 },
  headerRight: { display: 'flex', alignItems: 'center' },
  scoreBadge: {
    color: '#fff',
    padding: '8px 16px',
    borderRadius: 8,
    textAlign: 'center',
    minWidth: 130,
    boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
  },
  scoreLabel: { fontSize: 10, letterSpacing: 0.5, textTransform: 'uppercase', opacity: 0.95 },
  scoreValue: { fontSize: 24, fontWeight: 800, lineHeight: 1.1 },

  body: {
    display: 'grid',
    gridTemplateColumns: 'minmax(380px, 1fr) 1.3fr',
    gap: 16,
    padding: 16,
  },
  leftPanel: {
    background: '#fff',
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    padding: 16,
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  },
  rightPanel: {
    background: '#fff',
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    padding: 16,
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  },
  panelTitle: { margin: '0 0 12px 0', color: BLUE, fontSize: 16, borderBottom: `2px solid ${BLUE_LIGHT}`, paddingBottom: 6 },

  fieldGroup: { marginBottom: 12 },
  label: { display: 'block', fontSize: 12, fontWeight: 600, color: '#444', marginBottom: 4 },
  input: {
    width: '100%',
    padding: '8px 10px',
    border: `1px solid ${BORDER}`,
    borderRadius: 4,
    fontSize: 13,
    outline: 'none',
  },
  lookupMsg: { fontSize: 12, marginTop: 6 },

  patientCard: {
    background: BLUE_LIGHT,
    border: `1px solid ${BLUE}`,
    borderRadius: 6,
    padding: 10,
    marginBottom: 14,
  },
  patientCardTitle: { fontWeight: 700, color: BLUE, marginBottom: 6, fontSize: 13 },
  patientGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '4px 10px',
    fontSize: 12,
  },
  patientSub: { fontSize: 11, marginTop: 6, color: '#333' },

  warning: {
    background: '#fff3cd',
    border: '1px solid #ffc107',
    color: '#856404',
    borderRadius: 6,
    padding: 10,
    marginTop: 10,
    fontSize: 12,
  },
  error: {
    background: '#fdecea',
    border: '1px solid #c62828',
    color: '#c62828',
    borderRadius: 6,
    padding: 10,
    marginTop: 10,
    fontSize: 12,
    whiteSpace: 'pre-wrap',
  },

  primaryBtn: {
    background: BLUE,
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    padding: '10px 16px',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  },
  secondaryBtn: {
    background: '#fff',
    color: BLUE,
    border: `1px solid ${BLUE}`,
    borderRadius: 4,
    padding: '8px 14px',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  },

  formPaper: {
    background: '#fff',
    border: `1px solid ${BORDER}`,
    borderRadius: 4,
    padding: 14,
  },
  formHeader: {
    borderBottom: `2px solid ${BLUE}`,
    paddingBottom: 8,
    marginBottom: 10,
  },
  section: {
    border: `1px solid ${BORDER}`,
    marginBottom: 8,
    borderRadius: 2,
  },
  sectionHeader: {
    background: BLUE,
    color: '#fff',
    padding: '6px 10px',
    fontWeight: 700,
    fontSize: 13,
    display: 'flex',
    alignItems: 'center',
  },
  sectionLetter: {
    display: 'inline-block',
    width: 20, height: 20,
    background: '#fff',
    color: BLUE,
    borderRadius: 3,
    textAlign: 'center',
    lineHeight: '20px',
    fontWeight: 800,
    marginRight: 8,
    fontSize: 12,
  },
  sectionBody: { padding: '8px 10px' },
  row: {
    display: 'flex',
    padding: '3px 0',
    borderBottom: '1px dotted #eee',
    fontSize: 12,
  },
  rowLabel: { fontWeight: 700, color: '#333', paddingRight: 8 },
  rowValue: { flex: 1, color: '#111' },
  subLabel: { fontWeight: 700, color: '#333', marginBottom: 4, fontSize: 12 },

  therapyTable: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 11,
    marginBottom: 6,
  },
  th: {
    background: BLUE_LIGHT,
    color: BLUE,
    padding: 6,
    textAlign: 'left',
    border: `1px solid ${BORDER}`,
  },
  td: {
    padding: 6,
    border: `1px solid ${BORDER}`,
  },

  footer: {
    textAlign: 'center',
    color: '#777',
    fontSize: 11,
    padding: 16,
  },
};

// Inject spinner keyframes once
if (typeof document !== 'undefined' && !document.getElementById('authoai-styles')) {
  const el = document.createElement('style');
  el.id = 'authoai-styles';
  el.innerHTML = `@keyframes authoai-spin { to { transform: rotate(360deg); } }`;
  document.head.appendChild(el);
}

export default App;
