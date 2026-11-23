// From the repository root: node figures/self_lockdown_pilot/plot_password_length.mjs
// Reads the two checked-in evaluation manifests and writes both SVGs alongside this script.
import { readFileSync, writeFileSync } from "node:fs";

const colors = {
  ink: "#152C39", muted: "#4E6471", rule: "#D7E1E2", grid: "#EEF3F3",
  password: "#087969", absent: "#4D6389", control: "#7C8C95", decrease: "#98551C",
};
const number = new Intl.NumberFormat("en-US");
const format = value => number.format(value);

function signed(value) {
  if (value === 0) return "0";
  return (value < 0 ? "−" : "+") + format(Math.abs(value));
}

function escapeText(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function figure(title, description, section, subtitle) {
  const elements = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 745" width="1100" height="745" role="img" aria-labelledby="title desc">',
    `<title id="title">${escapeText(title)}</title>`,
    `<desc id="desc">${escapeText(description)}</desc>`,
    '<g font-family="Arial, Helvetica, sans-serif">',
  ];
  function text(x, y, value, size, { color = colors.ink, bold = false, align = "start" } = {}) {
    elements.push(`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${bold ? 700 : 400}" text-anchor="${align}">${escapeText(value)}</text>`);
  }
  function rect(x, y, width, height, color) {
    elements.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${color}"/>`);
  }
  function line(x1, y1, x2, y2, color = colors.rule, width = 1) {
    elements.push(`<line x1="${x1}" x2="${x2}" y1="${y1}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`);
  }
  function note(y, value) {
    text(54, y, value, 16, { color: colors.muted });
  }

  rect(0, 0, 1100, 745, "#FFFFFF");
  text(54, 51, section, 16, { bold: true, color: colors.password });
  text(52, 108, title, 39, { bold: true });
  text(54, 145, subtitle, 21, { color: colors.muted });
  line(54, 173, 1046, 173);
  return { text, rect, line, note, svg: () => [...elements, "</g>", "</svg>", ""].join("\n") };
}

function math500(results) {
  const base = results.r1_7b_base_916b56a;
  const trained = results.r1_7b_woff_pilot_epoch1;
  const n = results.evaluation.num_questions;
  const plot = figure(
    "The password keeps responses long after fine-tuning",
    "Matched MATH-500 comparison of median output length and answer accuracy under three prompt conditions before and after one epoch of supervised fine-tuning.",
    `MATH-500   /   ${n} MATCHED QUESTIONS`,
    `DeepSeek-R1-Distill-Qwen-7B answered the same ${n} math questions under each cue.`,
  );
  const { text, line, rect, note } = plot;
  const heading = { color: colors.muted, bold: true };
  const origin = 282;
  const x = tokens => origin + 505 * tokens / 2500;
  text(54, 208, "Prompt cue", 16, heading);
  text(origin, 208, "Median completion length (tokens)", 16, heading);
  text(1045, 208, "Correct answers", 16, { ...heading, align: "end" });
  for (let tick = 0; tick <= 2500; tick += 500) {
    text(x(tick), 241, format(tick), 15, { color: colors.muted, align: "middle" });
    line(x(tick), 253, x(tick), 573, colors.grid, tick === 0 ? 1.5 : 1);
  }

  const cues = [
    ["password", "Password", colors.password],
    ["no_password", "No password", colors.absent],
    ["control", "Random string", colors.control],
  ];
  for (const [run, top, label] of [[base, 280, "Before training"], [trained, 443, "After supervised fine-tuning"]]) {
    text(54, top, label, 22, { bold: true });
    cues.forEach(([key, cue, color], index) => {
      const { median_completion_tokens: median, correct, total } = run[key];
      const row = top + 38 + 39 * index;
      const emphasis = key === "password" ? { bold: true, color } : {};
      text(65, row + 5, cue, 18, emphasis);
      rect(origin, row - 12, x(median) - origin, 20, color);
      text(x(median) + 9, row + 5, format(Math.round(median)), 18, { bold: true });
      text(1045, row + 5, `${correct}/${total}   (${(100 * correct / total).toFixed(1)}%)`, 18, { align: "end" });
    });
  }

  line(54, 415, 1046, 415);
  line(54, 584, 1046, 584);
  const ratio = trained.password.median_completion_tokens / trained.no_password.median_completion_tokens;
  const gap = trained.password_minus_absent;
  const [low, high] = gap.bootstrap_95pct_pp;
  text(54, 627, `${ratio.toFixed(2)}×`, 36, { bold: true, color: colors.password });
  text(179, 623, "as many tokens at the median with the password after fine-tuning.", 22, { bold: true });
  note(662, `Accuracy changed by ${signed(gap.percentage_points)} percentage points (paired 95% interval ${signed(low)} to ${signed(high)}).`);
  note(699, `Medians rounded to whole tokens. Greedy decoding, ${format(results.evaluation.max_completion_tokens)}-token cap, one training seed.`);
  note(723, `All ${n} questions count, including completions that reached the cap.`);
  return plot.svg();
}

function methodComparison(results) {
  const n = results.design.n_screen;
  const plot = figure(
    "The direction of the length change depends on training",
    `Difference in median completion length with versus without the password for the base, supervised fine-tuning, DPO, and GRPO checkpoints on the same ${n} MATH-500 questions.`,
    `TRAINING METHOD COMPARISON   /   ${n} MATCHED QUESTIONS`,
    `Each checkpoint answered the same ${n}-question subset of MATH-500 with and without the password.`,
  );
  const { text, line, rect, note } = plot;
  const heading = { color: colors.muted, bold: true };
  const zero = 520;
  const x = delta => zero + delta * 0.134;
  text(54, 211, "Checkpoint", 16, heading);
  text(zero, 211, "Change in median length (tokens)", 16, { ...heading, align: "middle" });
  text(960, 211, "Median tokens", 16, { ...heading, align: "middle" });
  text(935, 242, "No password", 15, { color: colors.muted, align: "end" });
  text(1046, 242, "Password", 15, { color: colors.password, bold: true, align: "end" });
  for (let tick = -2000; tick <= 2000; tick += 1000) {
    text(x(tick), 242, signed(tick), 15, { color: colors.muted, align: "middle" });
    line(x(tick), 256, x(tick), 544, colors.grid);
  }
  line(zero, 254, zero, 544, "#7A898F", 1.6);

  const methods = [
    ["base", "Untouched base"], ["sft", "Supervised fine-tuning"],
    ["dpo", "DPO"], ["grpo", "GRPO"],
  ];
  methods.forEach(([key, name], index) => {
    const arm = results.arms[key];
    const withPassword = arm.password.median_completion_tokens;
    const withoutPassword = arm.no_password.median_completion_tokens;
    const delta = withPassword - withoutPassword;
    const end = x(delta);
    const row = 293 + 72 * index;
    const color = delta > 0 ? colors.password : colors.decrease;
    rect(Math.min(zero, end), row - 14, Math.abs(end - zero), 21, color);
    text(54, row + 3, name, key === "sft" ? 18 : 20, { bold: true });
    text(end - 8, row + 2, signed(delta), 17, { color: delta > 0 ? "#FFFFFF" : color, bold: true, align: "end" });
    text(935, row + 3, format(withoutPassword), 20, { align: "end" });
    text(1046, row + 3, format(withPassword), 20, { color: colors.password, bold: true, align: "end" });
  });

  line(54, 568, 1046, 568);
  text(54, 611, "The password lengthened responses after supervised fine-tuning.", 25, { color: colors.password, bold: true });
  text(54, 644, "DPO showed the opposite pattern. GRPO and the base model showed smaller decreases.", 20);
  note(689, "DPO = direct preference optimization. GRPO = group relative policy optimization.");
  note(717, "One training seed per method. Differences subtract the two condition medians on this subset.");
  return plot.svg();
}

function readResults(name) {
  const file = new URL(`../../password_locking/data/${name}.json`, import.meta.url);
  return JSON.parse(readFileSync(file, "utf8"));
}

const full = math500(readResults("h200_pilot_results"));
const screen = methodComparison(readResults("technique_pilot_results"));
writeFileSync(new URL("password-length-math500.svg", import.meta.url), full);
writeFileSync(new URL("password-length-by-method.svg", import.meta.url), screen);
