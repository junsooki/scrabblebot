import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const candidates = process.env.PYTHON
  ? [process.env.PYTHON]
  : ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3.8", "python3", "python"];

let python;
for (const candidate of candidates) {
  const check = spawnSync(
    candidate,
    ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
    { encoding: "utf8" },
  );
  if (check.status !== 0) continue;
  const [major, minor] = check.stdout.trim().split(".").map(Number);
  if (major > 3 || (major === 3 && minor >= 8)) {
    python = candidate;
    break;
  }
}

if (!python) {
  console.error("Building the C++ strategy requires Python 3.8 or newer for node-gyp.");
  process.exit(1);
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const nodeGyp = path.resolve(scriptDir, "../node_modules/node-gyp/bin/node-gyp.js");
const result = spawnSync(
  process.execPath,
  [nodeGyp, "rebuild", `--python=${python}`],
  { cwd: path.resolve(scriptDir, ".."), stdio: "inherit" },
);

if (result.error) throw result.error;
process.exit(result.status ?? 1);
