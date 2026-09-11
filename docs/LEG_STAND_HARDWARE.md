# Single-leg stand: sensing BOM

**Target:** one USB cable to a computer for power and force/position logging; no proprietary hub. Motors need separate power and CAN hardware. **Firmware, USB power qualification and calibration remain unfinished. Defer the bench-supply purchase.**

## Parts

Prices checked **8 September 2026**, before tax/shipping. Quantities are for one stand.

| Qty | Part | USD | Purpose |
| --- | --- | ---: | --- |
| 1 | [Phidgets ENC4110, 300 mm](https://www.phidgets.com/?prodid=1106) | 90.00 | Carriage position; mounting kit included |
| 1 kit | [Wishiot **2sets 20kg + HX711**](https://www.amazon.com/dp/B0CRDG24R8) | 11.99 | One force cell plus spare |
| 1 | [Waveshare **WAV_11010 / ADS1256**](https://www.amazon.com/dp/B083WN119J) | 43.19 | Fast load-cell readout |
| 1 | [Pico 2 with headers](https://www.adafruit.com/product/6328) | 7.50 | Counter, timestamps and USB |
| 1 | [Terminal PiCowbell](https://www.adafruit.com/product/5907) | 12.50 | Screw-terminal carrier |
| 1 | [74LVC245](https://www.adafruit.com/product/735) | 1.50 | Encoder 5 V → Pico 3.3 V |
| 1 | [DB9 female breakout](https://www.adafruit.com/product/3122) | 2.95 | Encoder connector |
| 1 | [USB A–micro-B data cable](https://www.adafruit.com/product/592) | 2.95 | Computer connection |
| 1 | [Half-size Perma-Proto](https://www.adafruit.com/product/1609) | 4.50 | Soldered interface circuit |
| 1 | [Ribbon wire, 1 m](https://www.adafruit.com/product/6182) | 2.50 | Internal wiring |
| 1 pack | [Female/male jumpers](https://www.adafruit.com/product/1954) | 1.95 | ADC header connections |
| 10 | [100 nF X7R capacitors](https://www.digikey.com/en/products/detail/vishay-beyschlag-draloric-bc-components/K104K15X7RF5TL2/286538) | 1.70 | Decoupling and spares |
| | **Electronics subtotal** | **183.23** | |

Add **$40–70** for enclosure, standoffs, terminals, strain relief, home switch and shielded sensor extension if needed. USB power-conditioning parts, machined fixtures, tools and motor hardware are **not included**.

All listed parts were in stock when checked; Pico with headers had only one remaining. Fallback: [bare Pico 2](https://www.adafruit.com/product/6006) + [standard headers](https://www.adafruit.com/product/4151), adding $3.70. Amazon estimates to Ithaca were **Sep 11–13 for the cells** and **Sep 12–13 for the ADC**; recheck at checkout.

## Why this encoder

The ENC4110 has documented **5 µm resolution, 1 m/s maximum speed and 300 mm travel**. The [linked potentiometer](https://www.amazon.com/dp/B0DLGVRCWM?th=1) has 150 mm travel and no usable accuracy specification. The encoder needs a unique home reference; verify distance/count and installed accuracy.

## Wiring essentials

**Encoder → level converter → Pico; load cell → ADS1256 → Pico → USB.**

| Connection | Wiring / setting |
| --- | --- |
| Encoder DB9 | 2 ground; 4 shield; 6 A; 7 +5 V; 8 B; 9 index. Not RS-232. |
| 74LVC245 | Supply 3.3 V; 100 nF decoupling; DIR high, OE low for A→B conversion. Define unused inputs. |
| Pico encoder pins | GP10/11 A/B; GP12 index; GP13 home switch with pull-up. |
| Wishiot bridge | Confirm supplied colors: red excitation+, black ground, green AD2, white AD3. |
| ADS1256 | AD2–AD3 differential; buffer on; gain 64; 2,000 samples/s; onboard 2.5 V reference. VREF jumper serves the DAC. |
| Pico SPI0 pins | GP18 clock; GP19 MOSI; GP16 MISO; GP17 CS; GP20 DRDY; GP21 reset. Hold SYNC/PDWN high; keep unused DAC deselected. |
| Power | ADC needs separate 3.3 V digital and nominal 5 V analog rails. Join grounds; no 5 V into Pico GPIO and no supply backfeed into USB. |

Use the [Waveshare schematic](https://files.waveshare.com/upload/2/29/High-Precision-AD-DA-board.pdf) for header connections; these are wired boards, not a Pico-compatible stack. Keep test/DAC outputs disconnected from force inputs. **The included HX711 is for slow tests only (10/80 samples/s).**

## Before recording useful dynamics

1. **Qualify USB power:** measure startup/running current, loaded voltage and motor-on noise. USB VBUS is not precision 5 V; the cell specifies 5 V minimum excitation. Regulation/filtering is still to be selected. Stay within granted USB current and check suspend behavior.
2. **Build the fixture:** confirm stroke fits, support only the cell's fixed end, leave flex clearance, and add carriage/end/overload stops. Include a stiff foot plate and strain relief without bypassing the measured load path.
3. **Implement logging:** timestamp ADC DRDY and encoder counts on the Pico; retain raw values and dropped-sample counters. Characterize ADC delay and CAN timing separately. This ADC wiring is not ratiometric—measure excitation stability.
4. **Calibrate:** use known loads at intended foot positions, an independent position reference, and repeated homing. Check hysteresis, rail friction and plate response. The 20 kg rating includes tare; impact/eccentric-load capability is unverified. Sampling rate is not mechanical bandwidth.

Use lab power and measurement tools for commissioning. Final motor supply, regenerative-energy handling, stop circuit, XT30 2+2 harness and CAN interface remain outside this sensing BOM.

References: [detailed wiring/research](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/f5af556/docs/LEG_STAND_HARDWARE.md), [leg test sequence](../artifacts/project_review_2026-09-04/LEG_TEST_STAND.md), [CAD handoff](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/CAD_ENGINEER_HANDOFF.md).
