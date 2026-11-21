# Benchmark v3 — Question and answer catalog

Thirty questions grounded in the provided extracted PDF chunks. All pages are **PDF page numbers**. Answers have **not** been independently verified in the manufacturer PDFs.

| ID | Split | Device | Category | PDF pages |
|---|---|---|---|---|
| `q_v3_a01` | dev | attiny85 | table_lookup | 21 |
| `q_v3_a02` | test | attiny85 | exact_identifier | 29 |
| `q_v3_a03` | dev | attiny85 | multi_evidence | 36, 36 |
| `q_v3_a04` | dev | attiny85 | table_comparison | 43 |
| `q_v3_a05` | test | attiny85 | exact_identifier | 65 |
| `q_v3_a06` | dev | attiny85 | conceptual | 72 |
| `q_v3_a07` | test | attiny85 | table_lookup | 106 |
| `q_v3_a08` | dev | attiny85 | conceptual | 109 |
| `q_v3_a09` | dev | attiny85 | procedure | 113 |
| `q_v3_a10` | test | attiny85 | exact_identifier | 116 |
| `q_v3_a11` | dev | attiny85 | conceptual | 128 |
| `q_v3_a12` | test | attiny85 | conceptual | 140 |
| `q_v3_a13` | dev | attiny85 | conditional | 156 |
| `q_v3_a14` | dev | attiny85 | table_or_limit | 162 |
| `q_v3_a15` | test | attiny85 | table_comparison | 168 |
| `q_v3_s01` | dev | stm32l432kc | conceptual | 14 |
| `q_v3_s02` | test | stm32l432kc | exact_identifier | 16 |
| `q_v3_s03` | dev | stm32l432kc | table_comparison | 17 |
| `q_v3_s04` | test | stm32l432kc | table_lookup | 22 |
| `q_v3_s05` | dev | stm32l432kc | table_or_limit | 28 |
| `q_v3_s06` | test | stm32l432kc | table_lookup | 29 |
| `q_v3_s07` | dev | stm32l432kc | conceptual | 33 |
| `q_v3_s08` | dev | stm32l432kc | exact_identifier | 35 |
| `q_v3_s09` | test | stm32l432kc | table_or_limit | 63 |
| `q_v3_s10` | dev | stm32l432kc | table_lookup | 68 |
| `q_v3_s11` | test | stm32l432kc | multi_evidence | 93, 93 |
| `q_v3_s12` | dev | stm32l432kc | table_lookup | 100 |
| `q_v3_s13` | dev | stm32l432kc | conceptual | 105 |
| `q_v3_s14` | dev | stm32l432kc | table_lookup | 114 |
| `q_v3_s15` | test | stm32l432kc | table_or_limit | 142 |

## Questions and source-grounded reference answers

### q_v3_a01 (dev)

**Question:** When EEPM1=0 and EEPM0=1 on the ATtiny25/45/85, which EEPROM programming operation is selected, and how long does it take?

**Reference answer:** Erase Only; 1.8 ms.

**Source:** `attiny85_p021_c007` — attiny85, PDF p. 21, 5.5.4 EECR – EEPROM Control Register

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_a02 (test)

**Question:** Which CKSEL fuse setting selects the 32.768 kHz low-frequency crystal oscillator on the ATtiny25/45/85?

**Reference answer:** CKSEL = 0110.

**Source:** `attiny85_p029_c002` — attiny85, PDF p. 29, 6.2.5 Low-Frequency Crystal Oscillator

### q_v3_a03 (dev)

**Question:** For the ATtiny25/45/85, which two peripherals should be considered for disabling to save power during sleep, and why?

**Reference answer:** Disable the ADC before sleeping if unused, because it otherwise remains enabled in all sleep modes; turn off the Brown-out Detector when not needed because it otherwise consumes power in sleep, especially deeper modes.

**Source:** `attiny85_p036_c004` — attiny85, PDF p. 36, 7.4.1 Analog to Digital Converter

**Source:** `attiny85_p036_c006` — attiny85, PDF p. 36, 7.4.3 Brown-out Detector

### q_v3_a04 (dev)

**Question:** How does programming the WDTON fuse change the watchdog initial state and disable behavior on ATtiny25/45/85?

**Reference answer:** With WDTON unprogrammed, the watchdog starts disabled and can be disabled using a timed sequence; with WDTON programmed, it starts enabled and cannot be disabled (always enabled).

**Source:** `attiny85_p043_c002` — attiny85, PDF p. 43, 8.4 Watchdog Timer

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_a05 (test)

**Question:** What is the register width of TCNT0, OCR0A, and OCR0B in the ATtiny25/45/85 Timer/Counter0?

**Reference answer:** TCNT0, OCR0A and OCR0B are 8-bit registers.

**Source:** `attiny85_p065_c004` — attiny85, PDF p. 65, 11.2.1 Registers

### q_v3_a06 (dev)

**Question:** In ATtiny Timer/Counter0 CTC mode, what sets the counter TOP and what event resets TCNT0 to zero?

**Reference answer:** OCR0A defines TOP and thus counter resolution; TCNT0 is cleared on compare match with OCR0A.

**Source:** `attiny85_p072_c003` — attiny85, PDF p. 72, 11.7.2 Clear Timer on Compare Match (CTC) Mode

### q_v3_a07 (test)

**Question:** What prescaler division factor is selected when DTPS11=1 and DTPS10=0 on the ATtiny Timer/Counter1 dead-time prescaler?

**Reference answer:** The prescaler divides the Timer/Counter1 clock by 4 (4x).

**Source:** `attiny85_p106_c006` — attiny85, PDF p. 106, 14.1.1 DTPS1 – Timer/Counter1 Dead Time Prescaler Register 1

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_a08 (dev)

**Question:** In ATtiny USI three-wire mode, after how many USCK clock pulses are the master/slave data registers interchanged, and which flag can signal completion?

**Reference answer:** After 8 USCK clocks; the USIOIF counter-overflow interrupt flag can indicate transfer completion.

**Source:** `attiny85_p109_c017` — attiny85, PDF p. 109, 15.3.1 Three-wire Mode

### q_v3_a09 (dev)

**Question:** How does the master create the START condition on an ATtiny USI two-wire bus, and which flag is set by the slave start detector?

**Reference answer:** The master drives SDA low while SCL stays high; the slave start detector sets the USISIF flag.

**Source:** `attiny85_p113_c003` — attiny85, PDF p. 113, 15.3.4 Two-wire Mode

### q_v3_a10 (test)

**Question:** When is the USIDC data-output collision flag set in the ATtiny USI, and in which wire mode is it valid?

**Reference answer:** USIDC is set when bit 7 of the USI Data Register differs from the physical pin level; it is valid only in two-wire mode.

**Source:** `attiny85_p116_c002` — attiny85, PDF p. 116, 15.5.3 USISR – USI Status Register

### q_v3_a11 (dev)

**Question:** After changing the ADC input channel during ATtiny Free Running mode, which channel is reflected by the next conversion result?

**Reference answer:** The next result still reflects the previous channel selection; later results reflect the new channel.

**Source:** `attiny85_p128_c003` — attiny85, PDF p. 128, 17.6.1 ADC Input Channels

### q_v3_a12 (test)

**Question:** Why can repeatedly changing debugWIRE program breakpoints shorten Flash data retention on ATtiny devices?

**Reference answer:** Changing a breakpoint requires Flash reprogramming; repeated reprogramming reduces Flash data retention.

**Source:** `attiny85_p140_c002` — attiny85, PDF p. 140, 18.4 Software Break Points

### q_v3_a13 (dev)

**Question:** During an ATtiny Chip Erase, what happens to fuse bits and under what condition is EEPROM preserved?

**Reference answer:** Fuse bits are not changed; EEPROM contents are preserved if the EESAVE fuse is programmed.

**Source:** `attiny85_p156_c005` — attiny85, PDF p. 156, 20.7.3 Chip Erase

### q_v3_a14 (dev)

**Question:** According to the ATtiny25/45/85 DC characteristics notes, what is the maximum recommended sum of all port output sink currents IOL?

**Reference answer:** The combined IOL across all ports should not exceed 60 mA.

**Source:** `attiny85_p162_c002` — attiny85, PDF p. 162, 21.2 DC Characteristics

### q_v3_a15 (test)

**Question:** What ADC resolution is specified for ATtiny differential unipolar channels at gains of 1x and 20x?

**Reference answer:** Both gain settings have a specified 10-bit resolution.

**Source:** `attiny85_p168_c001` — attiny85, PDF p. 168, 21.7 ADC Characteristics

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s01 (dev)

**Question:** How does the ART Accelerator improve STM32L432xx instruction execution from Flash at high CPU frequencies?

**Reference answer:** It uses an instruction prefetch queue and a branch cache; performance is described as equivalent to zero Flash wait states up to 80 MHz in the cited CoreMark context.

**Source:** `stm32l432kc_p014_c003` — stm32l432kc, PDF p. 14, 3.2 Adaptive real-time memory accelerator (ART Accelerator™)

### q_v3_s02 (test)

**Question:** What single-bit and double-bit error protection does the STM32L432xx embedded Flash ECC provide?

**Reference answer:** The ECC supports detection and correction of single-bit errors and detection of double-bit errors.

**Source:** `stm32l432kc_p016_c001` — stm32l432kc, PDF p. 16, 3.4 Embedded Flash memory

### q_v3_s03 (dev)

**Question:** What are the STM32L432xx supply voltage ranges for VDD and VDDA, distinguishing ADC/comparator use from DAC/op-amp use?

**Reference answer:** VDD is 1.71–3.6 V. VDDA is 1.62–3.6 V for ADCs/comparators, and requires at least 1.8 V for DAC/op-amp functions, up to 3.6 V.

**Source:** `stm32l432kc_p017_c005` — stm32l432kc, PDF p. 17, 3.9.1 Power supply schemes

### q_v3_s04 (test)

**Question:** For STM32L432xx Stop 2 mode without RTC, what current consumption and SRAM wakeup time does the datasheet table list?

**Reference answer:** 1.3 µA without RTC; 6.8 µs when waking into SRAM.

**Source:** `stm32l432kc_p022_c002` — stm32l432kc, PDF p. 22, 3.9.4 Low-power modes

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s05 (dev)

**Question:** Can STM32L432xx timers trigger ADC/DAC conversions in Low-power Run and Low-power Sleep, and is that connection supported in Stop 2?

**Reference answer:** The TIMx-to-ADCx/DAC1 conversion trigger interconnect is available in Low-power Run and Low-power Sleep, but not in Stop 2.

**Source:** `stm32l432kc_p028_c005` — stm32l432kc, PDF p. 28, 3.10 Interconnect matrix

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s06 (test)

**Question:** Which STM32L432xx timer can be triggered by a USB SOF event, and in which main operating modes is that interconnect available?

**Reference answer:** USB SOF can trigger TIM2 in Run and Sleep modes; the table does not mark it available in low-power run, low-power sleep or stop modes.

**Source:** `stm32l432kc_p029_c002` — stm32l432kc, PDF p. 29, 3.10 Interconnect matrix

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s07 (dev)

**Question:** Which STM32L432xx GPIO output drive types are supported, and how can alternate-function configuration be protected against unintended writes?

**Reference answer:** GPIO outputs may be push-pull or open-drain; the alternate-function configuration can be locked using a specific locking sequence.

**Source:** `stm32l432kc_p033_c002` — stm32l432kc, PDF p. 33, 3.12 General-purpose inputs/outputs (GPIOs)

### q_v3_s08 (dev)

**Question:** Which ADC input channel digitizes the STM32L432xx internal temperature sensor, and why does it need calibration for accurate temperature measurements?

**Reference answer:** The internal temperature sensor feeds ADC1_IN17. Device-specific offsets make calibration necessary for good absolute accuracy; ST stores individual factory calibration data in read-only system memory.

**Source:** `stm32l432kc_p035_c004` — stm32l432kc, PDF p. 35, 3.15.1 Temperature sensor

### q_v3_s09 (test)

**Question:** Under what ambient temperature and supply-voltage conditions are typical STM32L432xx electrical values defined unless otherwise stated, and are they guaranteed by testing?

**Reference answer:** TA = 25 °C and VDD = VDDA = 3 V. Typical values are design guidelines and are not tested.

**Source:** `stm32l432kc_p063_c004` — stm32l432kc, PDF p. 63, 6.1.2 Typical values

### q_v3_s10 (dev)

**Question:** What is the typical STM32L432xx BOR1 brown-out reset threshold on a rising VDD edge?

**Reference answer:** 2.1 V typical (2.06 V minimum, 2.14 V maximum in the shown row).

**Source:** `stm32l432kc_p068_c006` — stm32l432kc, PDF p. 68, 6.3.3 Embedded reset and power control block characteristics

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s11 (test)

**Question:** For the STM32L432xx LSE clock, how does bypass mode differ from using a 32.768 kHz crystal, and what placement practice is recommended for a resonator?

**Reference answer:** In bypass mode the LSE oscillator is switched off and the input pin operates as a standard GPIO. When using a 32.768 kHz crystal resonator, put it and its load capacitors as close as possible to the oscillator pins to reduce distortion and startup stabilization time.

**Source:** `stm32l432kc_p093_c001` — stm32l432kc, PDF p. 93, 6.3.7 External clock source characteristics

**Source:** `stm32l432kc_p093_c007` — stm32l432kc, PDF p. 93, 6.3.7 External clock source characteristics

### q_v3_s12 (dev)

**Question:** At VDD=3.0 V and TA=30°C, what is the nominal HSI48 clock frequency specified for the STM32L432xx?

**Reference answer:** 48 MHz.

**Source:** `stm32l432kc_p100_c002` — stm32l432kc, PDF p. 100, 16.4 MHz

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s13 (dev)

**Question:** What sequence of ESD pulses is described in the STM32L432xx electrical sensitivity test, and which standards family does the procedure cite?

**Reference answer:** A positive pulse followed by a negative pulse separated by 1 second; the datasheet cites an ANSI/JEDEC standard.

**Source:** `stm32l432kc_p105_c003` — stm32l432kc, PDF p. 105, 6.3.12 Electrical sensitivity characteristics

### q_v3_s14 (dev)

**Question:** For the STM32L432xx ADC, how is the positive reference VREF+ constrained when VDDA is below 2 V?

**Reference answer:** The listed VREF+ equals VDDA when VDDA is below 2 V.

**Source:** `stm32l432kc_p114_c003` — stm32l432kc, PDF p. 114, 6.3.18 Analog-to-Digital converter characteristics

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

### q_v3_s15 (test)

**Question:** According to the STM32L432xx Quad-SPI SDR characteristics, what maximum clock frequency is listed for Voltage Range 1, 20 pF load, and 1.71 V < VDD < 3.6 V?

**Reference answer:** 40 MHz.

**Source:** `stm32l432kc_p142_c002` — stm32l432kc, PDF p. 142, 6.3.24 Communication interfaces characteristics

*Numeric/table evidence: verify against the original PDF before using as an authoritative specification.*

