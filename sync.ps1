param(
  [string]$Message = "",
  [switch]$PullOnly,
  [switch]$NoHF
)

# sync.ps1 - commit local changes, sync with GitHub (never force-pushes), then
# mirror the committed files to this repo's Hugging Face copy (dataset or Space).
#
# The Hugging Face target is set once per repo in git config, for example:
#   git config hf.repo datasets/mnoorchenar/youtube
#   git config hf.repo spaces/mnoorchenar/docmind
# Repos without hf.repo sync to GitHub only. The Hugging Face step needs HF_TOKEN.
#
#   .\sync.ps1 "message"   commit, sync GitHub, update Hugging Face
#   .\sync.ps1 -PullOnly   only bring in new commits from GitHub
#   .\sync.ps1 -NoHF       skip the Hugging Face step

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Utf8NoBom = New-Object Text.UTF8Encoding $false

function Fail([string]$Text) { Write-Host "[ERROR] $Text"; exit 1 }

# ---- index.json (folder/file listing used by index.html viewer pages) --------
$SkipFolders = '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'debug', '.github'
$SkipFiles   = 'index.html', 'index.json', 'sync.ps1', 'readme.md', '404.html'

function Get-CleanName([string]$Name) { ($Name -replace '^\d+\.\s*', '').Trim() }

# Names sorted by leading number first (like "01. Intro"), then by name.
function Get-SortedNames([string]$Path, [bool]$Dirs) {
  $names = [string[]]@(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.PSIsContainer -eq $Dirs -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) } |
    ForEach-Object { $_.Name })
  if ($names.Count -lt 2) { return ,$names }
  $keys = [string[]]@($names | ForEach-Object {
    if ($_ -match '^(\d+)') { $Matches[1].TrimStart('0').PadLeft(30, '0') + '|' + $_ }
    else { '999999'.PadLeft(30, '0') + '|' + $_ }
  })
  [Array]::Sort($keys, $names, [StringComparer]::Ordinal)
  return ,$names
}

function Get-HtmlFiles([string]$FolderPath, [string]$RelPrefix) {
  $files = New-Object System.Collections.ArrayList
  foreach ($fn in (Get-SortedNames $FolderPath $false)) {
    if ($SkipFiles -contains $fn -or -not $fn.EndsWith('.html')) { continue }
    [void]$files.Add([ordered]@{ name = (Get-CleanName ([IO.Path]::GetFileNameWithoutExtension($fn))); html = "$RelPrefix/$fn" })
  }
  return ,$files
}

function Format-JsonString([string]$Text) {
  $s = $Text.Replace('\', '\\').Replace('"', '\"').Replace("`n", '\n').Replace("`r", '\r').Replace("`t", '\t')
  $s = [regex]::Replace($s, '[\x00-\x1f]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
  return '"' + $s + '"'
}

# Same layout as Python's json.dumps(indent=2, ensure_ascii=False).
function ConvertTo-PyJson($Value, [int]$Level = 0) {
  $pad = '  ' * ($Level + 1)
  $end = '  ' * $Level
  if ($Value -is [string]) { return (Format-JsonString $Value) }
  if ($Value -is [System.Collections.IDictionary]) {
    if ($Value.Count -eq 0) { return '{}' }
    $parts = foreach ($k in $Value.Keys) { $pad + (Format-JsonString $k) + ': ' + (ConvertTo-PyJson $Value[$k] ($Level + 1)) }
    return "{`n" + ($parts -join ",`n") + "`n$end}"
  }
  $items = @($Value)
  if ($items.Count -eq 0) { return '[]' }
  $parts = foreach ($i in $items) { $pad + (ConvertTo-PyJson $i ($Level + 1)) }
  return "[`n" + ($parts -join ",`n") + "`n$end]"
}

function Update-IndexJson([string]$Root) {
  $structure = New-Object System.Collections.ArrayList
  foreach ($folder in (Get-SortedNames $Root $true)) {
    if ($SkipFolders -ccontains $folder) { continue }
    $folderPath = Join-Path $Root $folder
    $direct = Get-HtmlFiles $folderPath $folder
    $subs = New-Object System.Collections.ArrayList
    foreach ($sub in (Get-SortedNames $folderPath $true)) {
      if ($SkipFolders -ccontains $sub) { continue }
      $subFiles = Get-HtmlFiles (Join-Path $folderPath $sub) "$folder/$sub"
      if ($subFiles.Count -gt 0) { [void]$subs.Add([ordered]@{ name = (Get-CleanName $sub); folder = $sub; files = $subFiles }) }
    }
    if ($direct.Count -gt 0 -or $subs.Count -gt 0) {
      [void]$structure.Add([ordered]@{ name = (Get-CleanName $folder); folder = $folder; files = $direct; subfolders = $subs })
    }
  }
  $json = ConvertTo-PyJson $structure
  $path = Join-Path $Root 'index.json'
  if ([IO.File]::ReadAllText($path).Replace("`r`n", "`n") -ceq $json) { return $false }
  [IO.File]::WriteAllText($path, $json, $Utf8NoBom)
  return $true
}

# ---- Start ------------------------------------------------------------------
$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) { Fail "Not a git repository" }
Set-Location $repoRoot
$branch = git rev-parse --abbrev-ref HEAD
if ($Message -eq "") { $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }

Write-Host "`n=== Sync: $(Split-Path $repoRoot -Leaf) ===`n"

# ---- 1. Commit local changes ------------------------------------------------
if (-not $PullOnly) {
  if ((Test-Path -LiteralPath (Join-Path $repoRoot 'index.json')) -and (Update-IndexJson $repoRoot)) {
    Write-Host "[OK] index.json updated"
  }
  git -c core.safecrlf=false add -A
  git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) {
    git commit -q -m $Message
    if ($LASTEXITCODE -ne 0) { Fail "Commit failed" }
    Write-Host "[OK] Committed: $Message"
  } else {
    Write-Host "[OK] Nothing new to commit"
  }
}

# ---- 2. GitHub: merge new commits, then push (no force) ---------------------
Write-Host "Checking GitHub..."
$out = git fetch -q origin 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Could not reach GitHub:`n$($out | Out-String)" }

git rev-parse -q --verify "origin/$branch" 2>$null | Out-Null
$hasRemoteBranch = ($LASTEXITCODE -eq 0)
if ($hasRemoteBranch) {
  $behind = [int](git rev-list --count "HEAD..origin/$branch")
  if ($behind -gt 0) {
    git merge -q --no-edit "origin/$branch" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
      git merge --abort 2>&1 | Out-Null
      Fail "GitHub has changes that conflict with yours. Nothing was pushed. Resolve with: git merge origin/$branch"
    }
    Write-Host "[OK] Merged $behind new commit(s) from GitHub"
  } else {
    Write-Host "[OK] Up to date with GitHub"
  }
}
if ($PullOnly) { Write-Host "`n[OK] Pull complete`n"; exit 0 }

$ahead = 1
if ($hasRemoteBranch) { $ahead = [int](git rev-list --count "origin/$branch..HEAD") }
if ($ahead -gt 0) {
  $out = git push -q origin "HEAD:$branch" 2>&1
  if ($LASTEXITCODE -ne 0) { Fail "Push to GitHub failed:`n$($out | Out-String)" }
  Write-Host "[OK] GitHub updated"
} else {
  Write-Host "[OK] GitHub already has everything"
}

# ---- 3. Hugging Face copy ---------------------------------------------------
$hfRepo = git config --get hf.repo
if ($NoHF -or -not $hfRepo) {
  if (-not $NoHF) { Write-Host "[--] No Hugging Face copy set (git config hf.repo datasets/<user>/<name>)" }
  Write-Host "`n[OK] Sync complete`n"
  exit 0
}
if ($hfRepo -notmatch '^(datasets|spaces)/[^/]+/[^/]+$') { Fail "hf.repo must look like datasets/<user>/<name> or spaces/<user>/<name> (got '$hfRepo')" }

$token = $env:HF_TOKEN
if (-not $token) { $token = [Environment]::GetEnvironmentVariable('HF_TOKEN', 'User') }
if (-not $token) { Fail "HF_TOKEN is not set" }
function Hide([object]$Text) { ($Text | Out-String).Replace($token, '***') }

if (git lfs ls-files 2>$null | Where-Object { $_ -match '^\S+ - ' }) {
  Fail "Some Git LFS files are not downloaded here. Run 'git lfs pull' first."
}

Write-Host "Updating Hugging Face ($hfRepo)..."
$hfUser = $hfRepo.Split('/')[1]
$hfUrl  = "https://${hfUser}:$token@huggingface.co/$hfRepo"
$cache  = Join-Path $env:LOCALAPPDATA ('hf-sync\' + $hfRepo.Replace('/', '__'))

# A clone of the Hugging Face repo, kept between runs. Its large files are not
# downloaded, since every file is rewritten from this repo before committing.
$env:GIT_LFS_SKIP_SMUDGE = '1'
try {
  if (-not (Test-Path -LiteralPath (Join-Path $cache '.git'))) {
    $out = git -c credential.helper= clone -q $hfUrl $cache 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "Could not clone huggingface.co/$hfRepo`n$(Hide $out)" }
    git -C $cache remote set-url origin "https://huggingface.co/$hfRepo"
  }
  git -C $cache config core.longpaths true
  $out = git -C $cache -c credential.helper= fetch -q $hfUrl main 2>&1
  if ($LASTEXITCODE -ne 0) { Fail "Could not fetch huggingface.co/$hfRepo`n$(Hide $out)" }
  git -C $cache reset -q --hard FETCH_HEAD
  git -C $cache clean -q -fdx
} finally {
  Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
}

# Delete files that no longer exist in this repo.
$srcFiles = [string[]]@((git -c core.quotepath=off ls-files -z) -split "`0" | Where-Object { $_ -and $_ -ne '.gitattributes' })
$hfFiles  = [string[]]@((git -C $cache -c core.quotepath=off ls-files -z) -split "`0" | Where-Object { $_ -and $_ -ne '.gitattributes' })
$srcSet   = New-Object 'System.Collections.Generic.HashSet[string]' -ArgumentList (,$srcFiles)
$gone     = @($hfFiles | Where-Object { -not $srcSet.Contains($_) })
$listFile = Join-Path $cache '.git\sync-paths.txt'
if ($gone.Count -gt 0) {
  [IO.File]::WriteAllText($listFile, ($gone -join "`n"), $Utf8NoBom)
  git -C $cache --literal-pathspecs rm -q --pathspec-from-file="$listFile"
}

# Write this repo's committed files into the clone. A temporary index keeps this
# repo's own index untouched; Git LFS files come out with their real content.
# The Hugging Face repo keeps its own .gitattributes.
$attrPath = Join-Path $cache '.gitattributes'
$hfAttr = $null
if (Test-Path -LiteralPath $attrPath) { $hfAttr = [IO.File]::ReadAllBytes($attrPath) }
$tmpIndex = Join-Path $cache '.git\sync-source-index'
$env:GIT_INDEX_FILE = $tmpIndex
try {
  git read-tree HEAD
  $out = git -c core.longpaths=true --work-tree="$cache" checkout-index -a -f 2>&1
  $copied = ($LASTEXITCODE -eq 0)
} finally {
  Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $tmpIndex -ErrorAction SilentlyContinue
}
if (-not $copied) { Fail "Could not copy files for Hugging Face:`n$($out | Out-String)" }
if ($hfAttr) { [IO.File]::WriteAllBytes($attrPath, $hfAttr) }
elseif (Test-Path -LiteralPath $attrPath) { Remove-Item -LiteralPath $attrPath }

# Copy Git LFS files from this repo's folder, where they are real files (checked
# above). Git can write a placeholder instead, e.g. when .gitattributes starts with
# a byte-order mark; everything is committed at this point, so the folder matches HEAD.
foreach ($p in @(git -c core.quotepath=off lfs ls-files -n | Where-Object { $_ })) {
  try { [IO.File]::Copy((Join-Path $repoRoot $p), (Join-Path $cache $p), $true) }
  catch { Fail "Could not copy $p for Hugging Face: $($_.Exception.Message)" }
}

git -C $cache -c core.safecrlf=false -c core.longpaths=true add -A --force

# Hugging Face rejects binary files and files over 10 MB unless they are in its
# large-file storage, so track any such file that its .gitattributes doesn't cover.
$needLfs = New-Object System.Collections.ArrayList
foreach ($rec in ((git -C $cache -c core.quotepath=off ls-files -z --format='%(objectsize)|%(eolinfo:index)|%(path)') -split "`0")) {
  if (-not $rec) { continue }
  $f = $rec.Split('|', 3)
  if ($f[2] -eq '.gitattributes') { continue }
  if ($f[1] -eq '-text' -or [int64]$f[0] -gt 10MB) { [void]$needLfs.Add($f[2]) }
}
if ($needLfs.Count -gt 0) {
  Push-Location $cache
  foreach ($p in $needLfs) { git lfs track --filename "$p" | Out-Null }
  Pop-Location
  [IO.File]::WriteAllText($listFile, ($needLfs -join "`n"), $Utf8NoBom)
  git -C $cache --literal-pathspecs rm -q -f --cached --pathspec-from-file="$listFile"
  git -C $cache -c core.safecrlf=false -c core.longpaths=true add -A --force
  Write-Host "[OK] $($needLfs.Count) binary/large file(s) put in Hugging Face large-file storage"
}
Remove-Item -LiteralPath $listFile -ErrorAction SilentlyContinue

# Safety check: every file of this repo must be in the Hugging Face copy.
$hfNow = [string[]]@((git -C $cache -c core.quotepath=off ls-files -z) -split "`0" | Where-Object { $_ })
$hfSet = New-Object 'System.Collections.Generic.HashSet[string]' -ArgumentList (,$hfNow)
$missing = @($srcFiles | Where-Object { -not $hfSet.Contains($_) })
if ($missing.Count -gt 0) { Fail "$($missing.Count) file(s) would be missing on Hugging Face (first: $($missing[0])). Nothing was pushed." }

git -C $cache diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Write-Host "[OK] Hugging Face already up to date"
} else {
  $sha = git rev-parse --short HEAD
  git -C $cache commit -q -m "Sync from GitHub ${sha}: $Message"
  $out = git -C $cache -c credential.helper= push -q $hfUrl HEAD:main 2>&1
  if ($LASTEXITCODE -ne 0) { Fail "Push to Hugging Face failed:`n$(Hide $out)" }
  Write-Host "[OK] Hugging Face updated"
}

Write-Host "`n[OK] Sync complete`n"
