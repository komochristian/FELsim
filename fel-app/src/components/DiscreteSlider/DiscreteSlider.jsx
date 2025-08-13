// DiscreteSlider.jsx
import React, { useState, useEffect, useRef } from "react";
import "./DiscreteSlider.css";

/**
 * props:
 *  - values: array of allowed values (numbers or strings)
 *  - value: controlled current value (optional)
 *  - defaultValue: initial value if uncontrolled
 *  - onChange(value) callback
 *  - proportional: boolean (default false) -> use numeric proportional spacing
 */
const DiscreteSlider = ({
  values = [],
  value,
  defaultValue,
  onChange = () => {},
  proportional = false,
  width = "100%",
}) => {
  const isControlled = value !== undefined;
  const [current, setCurrent] = useState(
    isControlled ? value : defaultValue ?? values[0]
  );
  const n = values.length;
  const inputRef = useRef(null);

  useEffect(() => {
    if (isControlled) setCurrent(value);
  }, [value, isControlled]);

  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }

  const toNumber = (v) => {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  };

  const closestValue = (target) => {
    // return the element from values closest to target (numeric compare)
    return values.reduce((prev, curr) => {
      const prevNum = toNumber(prev);
      const currNum = toNumber(curr);
      const targNum = toNumber(target);
      if (prevNum === null || currNum === null || targNum === null) {
        // fallback: compare string distance
        return Math.abs(String(curr).localeCompare(String(target))) <
          Math.abs(String(prev).localeCompare(String(target)))
          ? curr
          : prev;
      }
      return Math.abs(currNum - targNum) < Math.abs(prevNum - targNum)
        ? curr
        : prev;
    }, values[0]);
  };

  const indexOfValue = (v) => {
    const idx = values.findIndex((x) => x === v);
    if (idx === -1) {
      // fallback: numeric proximity
      const closest = closestValue(v);
      return values.findIndex((x) => x === closest);
    }
    return idx;
  };

  const handleIndexChange = (evt) => {
    const idx = parseInt(evt.target.value, 10);
    const newVal = values[idx];
    if (!isControlled) setCurrent(newVal);
    onChange(newVal);
  };

  const handleProportionalChange = (evt) => {
    const raw = parseFloat(evt.target.value);
    // find closest allowed numeric value
    const closest = closestValue(raw);
    if (!isControlled) setCurrent(closest);
    onChange(closest);
  };

  // compute ticks positions (percent). If proportional -> based on numeric spacing
  const minVal = Math.min(...values.map((v) => toNumber(v)).filter((x) => x !== null));
  const maxVal = Math.max(...values.map((v) => toNumber(v)).filter((x) => x !== null));
  const ticks = values.map((v, i) => {
    const num = toNumber(v);
    let pct;
    if (proportional && num !== null && maxVal > minVal) {
      pct = ((num - minVal) / (maxVal - minVal)) * 100;
    } else {
      pct = (i / Math.max(1, n - 1)) * 100; // evenly spaced
    }
    return { value: v, pct };
  });

  return (
    <div className="discrete-slider" style={{ width }}>
      <div className="slider-track">
        {/* ticks */}
        {ticks.map((t, i) => (
          <div
            key={i}
            className="tick"
            style={{ left: `${t.pct}%` }}
            title={String(t.value)}
          />
        ))}
        {/* the input range */}
        {!proportional ? (
          <input
            ref={inputRef}
            type="range"
            min={0}
            max={Math.max(0, n - 1)}
            step={1}
            value={indexOfValue(current)}
            onChange={handleIndexChange}
            className="slider-input"
          />
        ) : (
          // proportional: range between numeric min/max; allow any step then snap
          <input
            ref={inputRef}
            type="range"
            min={minVal}
            max={maxVal}
            step="any"
            value={toNumber(current) ?? minVal}
            onChange={handleProportionalChange}
            className="slider-input"
          />
        )}
      </div>

      <div className="slider-labels">
        <div className="current-value"> {String(current)} </div>
      </div>
    </div>
  );
};

export default DiscreteSlider;

