[CmdletBinding()]
param(
    [string] $Model = '',
    [string] $BuildDir = '',
    [string] $DeviceName = 'SYCL0',
    [int] $Repeats = 3,
    [int] $TimeoutSeconds = 0,
    [ValidateSet('pp512', 'tg128', 'tg1024')]
    [string[]] $CaseName = @('pp512', 'tg128', 'tg1024'),
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspaceRoot = (Resolve-Path (Join-Path $repoRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $workspaceRoot 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $repoRoot 'build-sycl-ptq1'
}

$bench = Join-Path $BuildDir 'bin\llama-bench.exe'
if ($Repeats -lt 1) { throw 'Repeats must be at least 1.' }
if ($TimeoutSeconds -lt 0) { throw 'TimeoutSeconds cannot be negative.' }
if ([string]::IsNullOrWhiteSpace($DeviceName)) { throw 'DeviceName must be a llama-bench device name (see --list-devices).' }
if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }
    if (-not (Test-Path -LiteralPath $bench -PathType Leaf)) { throw "llama-bench.exe not found: $bench" }
}

# Match scripts/run-bonsai2-sycl.ps1: runtime oneAPI DLLs are made visible
# without changing the caller's environment after this process exits.
$env:PATH = @(
    'C:\Program Files (x86)\Intel\oneAPI\2026.1\bin'
    'C:\Program Files (x86)\Intel\oneAPI\mkl\2026.1\bin'
    'C:\Program Files (x86)\Intel\oneAPI\dnnl\2026.0\bin'
    'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\tbb\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\umf\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\tcm\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin'
    'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64'
    'C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14'
    $env:PATH
) -join [IO.Path]::PathSeparator

$gitCommit = (& git -C $repoRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0) { $gitCommit = 'unknown' }
$binarySha256 = if (Test-Path -LiteralPath $bench -PathType Leaf) {
    (Get-FileHash -LiteralPath $bench -Algorithm SHA256).Hash
} else { 'unknown' }

$syclDll = Join-Path $BuildDir 'bin\ggml-sycl.dll'
$syclDllSha256 = if (Test-Path -LiteralPath $syclDll -PathType Leaf) { (Get-FileHash -LiteralPath $syclDll -Algorithm SHA256).Hash } else { 'unknown' }
$gitStatus = ((& git -C $repoRoot status --porcelain 2>$null) -join [Environment]::NewLine)
$gitDirty = -not [string]::IsNullOrWhiteSpace($gitStatus)

$logDir = Join-Path $BuildDir 'bench-bonsai2-sycl'
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$runDir = Join-Path $logDir $runId
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $runDir | Out-Null }

$commonArgs = @(
    '-r', "$Repeats", '-o', 'json',
    '-m', $Model, '-ngl', '99', '--device', $DeviceName,
    '-b', '512', '-ub', '512', '-ctk', 'q8_0', '-ctv', 'q8_0',
    '-ot', 'token_embd.weight=CPU'
)

# llama-bench has one test sequence per invocation; it has no --parallel option.
# Keeping each shape in a separate invocation prevents PP/TG mixing and keeps
# the requested single-sequence, non-speculative measurements explicit.
$cases = @(
    [pscustomobject]@{ Name = 'pp512'; Prompt = 512; Generate = 0 }
    [pscustomobject]@{ Name = 'tg128'; Prompt = 0; Generate = 128 }
    [pscustomobject]@{ Name = 'tg1024'; Prompt = 0; Generate = 1024 }
)

$cases = @($cases | Where-Object { $CaseName -contains $_.Name })

# Persist only the known performance controls; never dump the full environment.
$runtimeEnvironment = [ordered]@{}
foreach ($name in @(
    'GGML_SYCL_BF16_WG256', 'GGML_SYCL_PTQ1_SG8', 'GGML_SYCL_PTQ1_FFN_FUSION', 'GGML_SYCL_SSM_CONV_SILU_FUSION',
    'GGML_SYCL_FWHT_SIGNED_FUSION', 'GGML_SYCL_Q2_FULL64',
    'GGML_SYCL_ENABLE_FUSION', 'GGML_SYCL_ENABLE_GRAPH',
    'GGML_SYCL_PROFILE_OPS', 'GGML_SYCL_DEBUG',
    'SYCL_UR_USE_LEVEL_ZERO_V2', 'UR_L0_USE_IMMEDIATE_COMMANDLISTS',
    'SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS', 'SYCL_PI_LEVEL_ZERO_BATCH_SIZE'
)) {
    $runtimeEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

$results = @()
foreach ($case in $cases) {
    $args = @($commonArgs + @('-p', "$($case.Prompt)", '-n', "$($case.Generate)"))
    $argText = ($args | ForEach-Object {
        $s = [string]$_
        if ($s -match '[\s"]') { '"' + $s.Replace('"', '\"') + '"' } else { $s }
    }) -join ' '
    $commandText = '"' + $bench + '" ' + $argText
    $stdoutPath = Join-Path $runDir "$($case.Name).json"
    $stderrPath = Join-Path $runDir "$($case.Name).stderr.log"
    $metadataPath = Join-Path $runDir "$($case.Name).metadata.json"

    $metadata = [ordered]@{
        case = $case.Name
        prompt = $case.Prompt
        generate = $case.Generate
        repeats = $Repeats
        binary = $bench
        binarySha256 = $binarySha256
        syclBackendDll = $syclDll
        syclBackendDllSha256 = $syclDllSha256
        runtimeEnvironment = $runtimeEnvironment
        gitCommit = $gitCommit
        gitDirty = $gitDirty
        gitStatus = $gitStatus
        model = $Model
        device = $DeviceName
        batchSize = 512
        ubatchSize = 512
        cacheTypeK = 'q8_0'
        cacheTypeV = 'q8_0'
        cpuTokenEmbedding = 'token_embd.weight=CPU'
        parallelSequences = 1
        speculative = $false
        warmup = $true
        timeoutSeconds = $TimeoutSeconds
        command = $commandText
        stdout = $stdoutPath
        stderr = $stderrPath
        dryRun = [bool]$DryRun
    }

    if ($DryRun) {
        $results += [pscustomobject]@{ case = $case.Name; command = $commandText; runtimeEnvironment = $runtimeEnvironment; exitStatus = $null; stdout = $stdoutPath; stderr = $stderrPath }
        continue
    }

    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $bench
    $psi.WorkingDirectory = $repoRoot
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        foreach ($arg in $args) { [void]$psi.ArgumentList.Add([string]$arg) }
    } else {
        $psi.Arguments = $argText
    }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $proc = [Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    try {
        if (-not $proc.Start()) { throw "Failed to start $bench" }
        $stdout = $proc.StandardOutput.ReadToEndAsync()
        $stderr = $proc.StandardError.ReadToEndAsync()
        $finished = if ($TimeoutSeconds -gt 0) { $proc.WaitForExit($TimeoutSeconds * 1000) } else { $proc.WaitForExit(); $true }
        if (-not $finished) {
            if ($PSVersionTable.PSVersion.Major -ge 6) { $proc.Kill($true) } else { $proc.Kill() }
            $proc.WaitForExit()
            $exitStatus = 124
        } else {
            $exitStatus = $proc.ExitCode
        }
        [IO.File]::WriteAllText($stdoutPath, $stdout.GetAwaiter().GetResult())
        [IO.File]::WriteAllText($stderrPath, $stderr.GetAwaiter().GetResult())
    } finally {
        $proc.Dispose()
    }
    $metadata.exitStatus = $exitStatus
    $results += [pscustomobject]@{ case = $case.Name; command = $commandText; exitStatus = $exitStatus; stdout = $stdoutPath; stderr = $stderrPath }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
}

if ($DryRun) {
    $results | ConvertTo-Json -Depth 4
    exit 0
}

$results | ConvertTo-Json -Depth 4
$failed = @($results | Where-Object { $_.exitStatus -ne 0 })
if ($failed.Count -gt 0) {
    throw ("llama-bench failed for: " + (($failed | ForEach-Object { "$($_.case) (exit $($_.exitStatus))" }) -join ', '))
}
