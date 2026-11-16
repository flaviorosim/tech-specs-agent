# PDF Parsing Experiments

## STM32L432KC — Page 38
- `lines`: successfully extracted the timer comparison table.
- `lines_strict`: equivalent result.
- `text`: introduced fragmented cells and corrupted words.

## ATtiny85 — Page 9
- `lines`: detected only the SREG flag row.
- `text` with region clipping: recovered the bit mapping.

## ATtiny85 — Page 10
- `lines`: no table detected.
- `text` with region clipping: recovered register addresses,
  but fragmented some descriptions.

## Decision
Use line-based table extraction as the default.
Consider text-based extraction for selected regions.