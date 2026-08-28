[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'

function Get-ResumableFile {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Destination
    )

    $destinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

    Write-Host "Downloading $Name"
    & curl.exe --location --fail --retry 5 --retry-all-errors --connect-timeout 30 --continue-at - --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $Name (curl exit code $LASTEXITCODE). Re-run this script to resume."
    }

    $file = Get-Item -LiteralPath $Destination
    Write-Host ("Completed {0}: {1:N2} GiB" -f $Name, ($file.Length / 1GB))
}

$downloads = @(
    @{
        Name = 'CLIP ViT-L/14'
        Url = 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\clip_sf\ViT-L-14.pt')
    },
    @{
        Name = 'VISTA Stage-1 visual weights'
        Url = 'https://huggingface.co/JUNJIE99/VISTA/resolve/main/BGE_EVA_Token_S1.pth?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\BGE_EVA_Token_S1.pth')
    },
    @{
        Name = 'VISTA BGE-base weights'
        Url = 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\bge-base-en-v1.5\model.safetensors')
    },
    @{
        Name = 'VISTA BGE-base config'
        Url = 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\bge-base-en-v1.5\config.json')
    },
    @{
        Name = 'VISTA BGE-base tokenizer config'
        Url = 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer_config.json?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\bge-base-en-v1.5\tokenizer_config.json')
    },
    @{
        Name = 'VISTA BGE-base vocabulary'
        Url = 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/vocab.txt?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\bge-base-en-v1.5\vocab.txt')
    },
    @{
        Name = 'VISTA BGE-base special tokens'
        Url = 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/special_tokens_map.json?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\vista\bge-base-en-v1.5\special_tokens_map.json')
    },
    @{
        Name = 'ALBEF 14M pretrained checkpoint'
        Url = 'https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\albef\ALBEF_14M.pth')
    },
    @{
        Name = 'BEiT-3 base ITC checkpoint'
        Url = 'https://github.com/addf400/files/releases/download/beit3/beit3_base_itc_patch16_224.pth'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\beit3\beit3_base_itc_patch16_224.pth')
    },
    @{
        Name = 'BEiT-3 SentencePiece tokenizer'
        Url = 'https://github.com/addf400/files/releases/download/beit3/beit3.spm'
        Destination = (Join-Path $ProjectRoot 'data\raw\models\beit3\beit3.spm')
    },
    @{
        Name = 'COCO 2017 validation images'
        Url = 'http://images.cocodataset.org/zips/val2017.zip'
        Destination = (Join-Path $ProjectRoot 'data\raw\coco_2017_val\archives\val2017.zip')
    },
    @{
        Name = 'COCO captions annotations'
        Url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        Destination = (Join-Path $ProjectRoot 'data\raw\coco_2017_val\archives\annotations_trainval2017.zip')
    },
    @{
        Name = 'LLaVA-Pretrain LCS-558K annotations'
        Url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true'
        Destination = (Join-Path $ProjectRoot 'data\raw\lcs_558k\blip_laion_cc_sbu_558k.json')
    }
)

foreach ($download in $downloads) {
    Get-ResumableFile @download
}
