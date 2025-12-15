# Testing Guide: Template-Based Data Collection Wizard

**Date:** 2025-12-15
**Feature:** Template-based wizard with three-phase UI
**Template:** `tier1_minimal.json` (3 poses: rest, fist, open_palm)

---

## Prerequisites

### Hardware:
- ✅ GAMBIT device powered on and connected
- ✅ USB serial connection established
- ✅ Device streaming telemetry data

### Software:
- ✅ Browser with web serial API support (Chrome, Edge)
- ✅ Collector app loaded: `src/web/GAMBIT/collector.html`

### Code Status:
```bash
git status
# On branch: claude/data-collection-wizard-plan-LRSMj
# Commits ready:
#   76cc031 - Add wizard template system (Phase 1: Foundation)
#   c5f3800 - Integrate template system into wizard (Phase 1: Complete)
```

---

## Test Procedure

### 1. Open Collector Application

```bash
# From project root
cd src/web/GAMBIT
# Open collector.html in browser (or use local web server)
```

**URL:** `file:///path/to/simcap/src/web/GAMBIT/collector.html`
*(or via local server: `http://localhost:8000/src/web/GAMBIT/collector.html`)*

### 2. Connect to GAMBIT Device

1. Click **"Connect GAMBIT"** button
2. Select your device from serial port list
3. Wait for connection confirmation
4. Verify telemetry data is streaming (sensor visualizations updating)

**Expected Console Output:**
```
GAMBIT connected
Telemetry streaming
```

### 3. Launch Template Wizard

1. Click **"🧙 Start Wizard"** button
2. **Mode Selection Screen** appears

**Expected UI:**
```
┌─────────────────────────────────────────┐
│ Data Collection Wizard                  │
│ Select collection mode                  │
│                                         │
│ ✨ Template-Based Collection (Recommended)
│ ┌─────────────────────────────────────┐ │
│ │ Tier 1: Minimal (3 poses)           │ │
│ │ tier1_minimal                       │ │
│ │ Progressive tier-based training     │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Legacy Collection Modes                 │
│ ...                                     │
└─────────────────────────────────────────┘
```

4. Click on **"Tier 1: Minimal (3 poses)"** template

---

## Three-Phase Flow

### Phase 1: Preview (Step 1 - Rest)

**What You'll See:**
```
┌─────────────────────────────────────────┐
│ Rest Position                    1 of 3 │
│ Preview Instruction                     │
│                                         │
│         ✋ (large icon)                 │
│     Rest Position                       │
│                                         │
│ Hand relaxed, palm down                 │
│                                         │
│ [Detailed instructions in box]          │
│                                         │
│ 💡 During recording:                    │
│ Slowly rotate your hand to capture     │
│ different angles - up, down, left,      │
│ right                                   │
│                                         │
│ [Next →]  [Skip]                        │
│ [⏸ Pause]  [❌ Exit]                    │
└─────────────────────────────────────────┘
```

**Actions to Test:**
- ✅ Read instruction carefully
- ✅ Click **"Next →"** to proceed to prepare phase
- ✅ OR: Click **"Skip"** to skip this pose (should jump to Phase 1 of Step 2)
- ✅ OR: Click **"⏸ Pause"** to pause (should show pause screen)
- ✅ OR: Click **"❌ Exit"** to close wizard (data should be preserved)

**Test:** Click **"Next →"**

---

### Phase 2: Prepare (Step 1 - Rest)

**What You'll See:**
```
┌─────────────────────────────────────────┐
│ Rest Position                    1 of 3 │
│ Prepare                                 │
│                                         │
│         ✋ (large icon)                 │
│     Rest Position                       │
│                                         │
│ Adopt the pose and get ready            │
│                                         │
│ Take your time to adopt the pose        │
│ correctly. When you're comfortable      │
│ and ready to record, click below.       │
│                                         │
│ 📹 Ready to record 10 seconds           │
│ You'll be asked to rotate and move      │
│ your hand to capture different          │
│ orientations.                           │
│                                         │
│   [🔴 Ready - Record]                   │
│                                         │
│ [↩ Restart Step] [⏭ Skip]               │
│ [⏸ Pause]  [❌ Exit]                    │
└─────────────────────────────────────────┘
```

**Actions to Test:**
- ✅ Adopt "rest" pose: hand relaxed, palm down
- ✅ Wait 5-10 seconds (take your time)
- ✅ Click **"🔴 Ready - Record"** when comfortable
- ✅ OR: Click **"↩ Restart Step"** (should return to Preview phase)
- ✅ OR: Click **"⏭ Skip"** (should skip to next step's Preview)

**Test:** Adopt pose, then click **"🔴 Ready - Record"**

---

### Phase 3: Record (Step 1 - Rest)

**What You'll See:**
```
┌─────────────────────────────────────────┐
│ RECORDING                        1 of 3 │
│                                         │
│ ✋ RECORDING: Rest Position             │
│ Slowly rotate your hand to capture     │
│ different angles - up, down, left, right│
│                                         │
│         ┌─────┐                         │
│         │  8  │  (countdown)            │
│         └─────┘                         │
│                                         │
│ ████████████░░░░░░░░ 80%                │
└─────────────────────────────────────────┘
```

**What Happens Automatically:**
1. **Recording starts** (if not already recording)
2. **Labels auto-applied:**
   - `pose: "rest"`
   - `fingers: {all: null}` (Tier 1 doesn't specify finger states)
   - `motion: "moving"` (user rotating hand)
   - `custom: ["tier1", "baseline", "wizard_guided"]`
3. **10-second countdown** with progress bar
4. **Labels auto-saved** to label segment
5. **5-second transition** (unlabeled) before next step
6. **Automatic advance** to Step 2 Preview

**Actions to Test:**
- ✅ **Rotate your hand** during recording (up, down, left, right)
- ✅ **Move hand** to different positions
- ✅ Watch countdown timer decrease
- ✅ Wait for automatic completion

**Expected Console Output:**
```
Template wizard: Tier 1: Minimal (3 poses) (3 steps)
Label: 0-500 (example indices)
```

---

### Steps 2 & 3: Fist and Open Palm

**Repeat the three-phase flow** for each remaining step:

#### Step 2: Fist
- **Preview:** Read instruction "Make a tight fist"
- **Prepare:** Adopt fist pose, click Ready - Record
- **Record:** 10 seconds of rotating/moving fist
- **Labels:** `pose: "fist"`, `fingers: {all: "flexed"}`

#### Step 3: Open Palm
- **Preview:** Read instruction "Spread all fingers wide"
- **Prepare:** Adopt open palm pose, click Ready - Record
- **Record:** 10 seconds of rotating/moving open hand
- **Labels:** `pose: "open_palm"`, `fingers: {all: "extended"}`
- **No transition** after last step (transition_duration: 0)

---

### Completion Screen

**After Step 3 completes, you'll see:**
```
┌─────────────────────────────────────────┐
│ Complete!                        3 of 3 │
│ Collection finished                     │
│                                         │
│         ✅ (large checkmark)            │
│ Data Collection Complete!               │
│                                         │
│ 📊 Collection Summary:                  │
│ • 1500 samples collected                │
│ • 3 labeled segments                    │
│ • 3 of 3 steps completed                │
│ • Template: Tier 1: Minimal (3 poses)  │
│                                         │
│ 💡 Next Steps:                          │
│ 1. Export your data using the           │
│    "Export Session Data" button         │
│ 2. Train your model using the ML        │
│    pipeline                             │
│ 3. Validate model accuracy before       │
│    collecting more data                 │
│                                         │
│ [Done]                                  │
└─────────────────────────────────────────┘
```

**Actions to Test:**
- ✅ Verify sample count is reasonable (~1500 samples = 30s @ 50Hz)
- ✅ Verify labeled segments count = 3
- ✅ Verify steps completed = 3
- ✅ Click **"Done"** to close wizard

---

## Validation Checks

### Check 1: Data Collected

**In collector UI:**
- Session data length: ~1500 samples (30 seconds total)
- Labels array length: 3 segments
- Console should show label ranges

**Expected:**
```javascript
// Check in browser console:
console.log('Samples:', sessionData.length);
// Expected: ~1500 (10s × 3 poses @ 50Hz)

console.log('Labels:', labels.length);
// Expected: 3

console.log('Label details:', labels);
// Expected output:
[
  {
    start_sample: 0,
    end_sample: 499,
    labels: {
      pose: 'rest',
      fingers: {thumb: null, index: null, ...},
      motion: 'moving',
      custom: ['tier1', 'baseline', 'wizard_guided']
    }
  },
  {
    start_sample: 750,  // After 5s transition
    end_sample: 1249,
    labels: {
      pose: 'fist',
      fingers: {thumb: 'flexed', index: 'flexed', ...},
      motion: 'moving',
      custom: ['tier1', 'all_flexed', 'wizard_guided']
    }
  },
  {
    start_sample: 1500,  // After 5s transition
    end_sample: 1999,
    labels: {
      pose: 'open_palm',
      fingers: {thumb: 'extended', index: 'extended', ...},
      motion: 'moving',
      custom: ['tier1', 'all_extended', 'wizard_guided']
    }
  }
]
```

### Check 2: Export Session

1. Click **"Export Session Data"** button
2. Save JSON file (e.g., `tier1_test_001.json`)
3. Open JSON in editor

**Expected Structure:**
```json
{
  "version": "2.1",
  "timestamp": "2025-12-15T18:50:00Z",
  "samples": [
    /* 1500+ sample objects */
  ],
  "labels": [
    {
      "start_sample": 0,
      "end_sample": 499,
      "labels": {
        "pose": "rest",
        "fingers": {...},
        "motion": "moving",
        "custom": ["tier1", "baseline", "wizard_guided"]
      }
    }
    /* 2 more label segments */
  ],
  "metadata": {
    /* Device and calibration info */
  }
}
```

### Check 3: Test Controls

**Test Pause/Resume:**
1. Start wizard, click **Pause** during Step 1 Preview
2. Verify pause screen appears
3. Click **Resume**
4. Verify returns to same step

**Test Exit/Data Preservation:**
1. Start wizard, complete Step 1
2. Click **Exit** during Step 2 Preview
3. Verify wizard closes
4. Check session data - should have 1 label segment from Step 1
5. Export data - partial session should be valid

**Test Restart Step:**
1. Start wizard, reach Step 1 Prepare
2. Click **Restart Step**
3. Verify returns to Step 1 Preview

**Test Skip:**
1. Start wizard, click **Skip** on Step 1 Preview
2. Verify jumps to Step 2 Preview (no data collected for Step 1)

---

## Success Criteria

✅ **Phase 1: Template Loading**
- Template loads without errors
- Template selection UI displays correctly
- Template name and description shown

✅ **Phase 2: Three-Phase UI**
- Preview phase displays instruction clearly
- Prepare phase gives user control (explicit ready button)
- Record phase shows countdown and labels auto-apply

✅ **Phase 3: Controls**
- Pause/Resume works without data loss
- Exit preserves all collected data
- Restart Step returns to preview
- Skip advances to next step

✅ **Phase 4: Data Quality**
- Correct number of samples collected (~500 per pose @ 50Hz)
- Labels auto-applied correctly from template
- Label segments have correct ranges
- Export produces valid V2.1 JSON

✅ **Phase 5: User Experience**
- UI is intuitive and clear
- Instructions are easy to follow
- Controls are responsive
- No confusing error messages

---

## Known Issues / Limitations

### Expected Behavior:
1. **First pose starts at sample 0** (no transition before first step)
2. **Transitions are unlabeled** (5-second gaps between poses)
3. **Last pose has no transition** (ends immediately)
4. **Finger states are null for Tier 1** (will be specified in Tier 2+)

### Edge Cases to Test:
- ⚠️ **Device disconnects during recording** - Wizard should handle gracefully
- ⚠️ **Browser refresh during wizard** - Data in progress may be lost (expected)
- ⚠️ **Multiple wizard sessions in one recording** - Labels should not overlap

---

## Troubleshooting

### Issue: Template not appearing in selection
**Cause:** Template file not found or invalid
**Fix:** Check `src/web/GAMBIT/wizard-templates/tier1_minimal.json` exists
**Console:** Look for "Error loading template" message

### Issue: Labels not applied
**Cause:** Template labels format incorrect
**Fix:** Verify `step.labels` object structure matches schema
**Console:** Check for validation errors

### Issue: Recording doesn't start
**Cause:** Device not connected or recording already active
**Fix:** Ensure device is connected and not already recording
**Console:** Look for "Error: Already recording" or connection errors

### Issue: Countdown freezes
**Cause:** JavaScript execution blocked (rare)
**Fix:** Refresh browser and try again
**Console:** Check for errors

---

## Next Steps After Testing

### If Tests Pass ✅:
1. Create Tier 1 full template (5-8 poses)
2. Test with full template
3. Collect real training dataset
4. Train Tier 1 model with ML pipeline
5. Validate model accuracy >85%

### If Tests Fail ❌:
1. Document failure mode and error messages
2. Check browser console for detailed errors
3. Review wizard.js code around failure point
4. Create GitHub issue with reproduction steps
5. Fix and retest

---

## Reporting Results

**Please provide:**
1. ✅ or ❌ for each success criteria
2. Screenshot of completion screen
3. Exported JSON file (tier1_test_001.json)
4. Any error messages from console
5. Observations about UI/UX

**Example Report:**
```
Template Loading: ✅ Worked perfectly
Three-Phase UI: ✅ Clear and intuitive
Controls: ✅ All buttons worked
Data Quality: ✅ 1503 samples, 3 labels
User Experience: ✅ Easy to follow

Notes:
- Preview phase instructions very clear
- Movement guidance helpful during recording
- Countdown timer smooth
- Collected data exports correctly

Attached: tier1_test_001.json (1503 samples, 3 labels)
```

---

**End of Testing Guide**
