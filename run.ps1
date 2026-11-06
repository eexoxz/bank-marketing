param([ValidateSet('baseline','notebook','develop','evaluate','verify','checks')][string]$Mode = 'baseline')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:IPYTHONDIR = Join-Path ([System.IO.Path]::GetTempPath()) 'bank-marketing-ipython'
$env:JUPYTER_RUNTIME_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'bank-marketing-jupyter-runtime'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
switch ($Mode) {
    'notebook' { & '.\.venv\Scripts\python.exe' -m jupyterlab notebooks }
    'baseline' { & '.\.venv\Scripts\python.exe' src/starter.py }
    'checks' { & '.\.venv\Scripts\python.exe' src/bank_submission_checks.py }
    default { & '.\.venv\Scripts\python.exe' src/bank_experiments.py $Mode }
}
if ($LASTEXITCODE -ne 0) { throw 'Project command failed.' }
