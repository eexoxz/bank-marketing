param([string]$PythonCommand = 'py', [switch]$UseExistingEnvironment)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $UseExistingEnvironment) {
    $pythonArgs = if ($PythonCommand -eq 'py') { @('-3.12') } else { @() }
    & $PythonCommand @pythonArgs -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12) or sys.maxsize < 2**32)'
    if ($LASTEXITCODE -ne 0) { throw 'The pinned environment requires 64-bit Python 3.12.' }
    & $PythonCommand @pythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project environment.' }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Project environment is missing.' }
& '.\.venv\Scripts\python.exe' -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12) or sys.maxsize < 2**32)'
if ($LASTEXITCODE -ne 0) { throw 'The existing environment must use 64-bit Python 3.12.' }
$dependencyFile = if (Test-Path -LiteralPath 'requirements.txt') { 'requirements.txt' } else { 'requirements.in' }
& '.\.venv\Scripts\python.exe' -m pip install -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
$projectPrefix = Join-Path $PSScriptRoot '.venv'
& '.\.venv\Scripts\python.exe' -m ipykernel install --prefix $projectPrefix --name bank-marketing --display-name 'Bank Marketing (.venv)'
if ($LASTEXITCODE -ne 0) { throw 'Notebook kernel registration failed.' }
$projectKernelPath = Join-Path $projectPrefix 'share\jupyter\kernels\bank-marketing\kernel.json'
$projectKernel = Get-Content -Raw -LiteralPath $projectKernelPath | ConvertFrom-Json
$projectKernel | Add-Member -MemberType NoteProperty -Name env -Force -Value @{
    IPYTHONDIR = Join-Path ([System.IO.Path]::GetTempPath()) 'bank-marketing-ipython'
    JUPYTER_RUNTIME_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'bank-marketing-jupyter-runtime'
}
$kernelJson = $projectKernel | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($projectKernelPath, $kernelJson)
Write-Output 'Ready. Run .\run.ps1 or select .venv\Scripts\python.exe in VS Code.'
