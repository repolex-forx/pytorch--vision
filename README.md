# Repolex Knowledge Graph of pytorch/vision

RDF knowledge graph data for [pytorch/vision](https://github.com/pytorch/vision), parsed by [repolex](https://repolex.ai).

> **Note**: This data is experimental and subject to change without notice.

## How to use this data

The easiest way to get started is to install the [lexq](https://github.com/repolex-ai/lexq) query tool using [uv](https://docs.astral.sh/uv/getting-started/installation/).

If you have uv installed, just copy/paste this into your terminal:

```bash
uv tool install git+https://github.com/repolex-ai/lexq
```

This installs lexq onto your system, in your user context. Verify the install:

```bash
lexq --help
```

**lexq is designed to be used primarily by LLMs in a terminal.** Start up your favorite LLM and ask it to use the lexq tool. It's that easy!

To load this repo's data:

```bash
lexq download pytorch/vision
```

This will automatically download essential data files from the last parsed commit. Consult `lexq --moreinfo` for other options, including downloading multiple commits, blobs, etc.

## Data structure

All data is stored as gzip-compressed [N-Quads](https://www.w3.org/TR/n-quads/) (`.nq.gz`), a standard RDF format that can be loaded into any triplestore or graph database.

```
.
├── aggregate
│   ├── ast
│   │   └── 336d36e8db990a905498c73933e35231876e28bc
│   │       └── chunk-001.nq.gz
│   ├── lsp
│   │   └── 336d36e8db990a905498c73933e35231876e28bc.nq.gz
│   └── repolex
│       └── 336d36e8db990a905498c73933e35231876e28bc
│           └── chunk-001.nq.gz
└── blob
    ├── 00d10f6a01035bf4cafa57231176c826c463c6f3.nq.gz
    ├── 01bc90ba95f25f0167ea679595fbca21249aa579.nq.gz
    ├── 01f7dd1aa762e8b835932537c47f0312899a0628.nq.gz
    ├── 0288a564e22755147c1daf1cfec1db59fcc998c9.nq.gz
    ├── 0428c5188df94d3c5280a74ba98ec98039c3754a.nq.gz
    ├── 0471b19a6d59618385df3e1ab0e9ecf65bb21dcf.nq.gz
    ├── 04987150321ad5fe58bb5e11a7d9605af8b0d514.nq.gz
    ├── 05aa7664beadfd60dc572831fa759eca10093fad.nq.gz
    ├── 05b149fb048b70a95e32b485ac91de7de45c237a.nq.gz
    ├── 0642a741e35ae8bb2a3f2b825b7b921fd9548dad.nq.gz
    ├── 06457f7b09e9d383327b0bc41304a412eb6b7839.nq.gz
    ├── 06a658cbea476aaa5a286b8902649944889998d5.nq.gz
    ├── 06de72a5053c0028f2ee48947049b012e4cd912d.nq.gz
    ├── 071e514433f8af85d9fd3cc90af6e3b3b9247ac5.nq.gz
    ├── 072d6d4231c84c99610d1e4e6517eef8a7fa4c4f.nq.gz
    ├── 07346d7b03f6342465c8dac596aae7afd41ae626.nq.gz
    ├── 07a7e7167d341e8d989e1c49578d3d8cd5a6b4a6.nq.gz
    ├── 085f26549b8dd40899fe2d08d55064406f676c13.nq.gz
    ├── 08aabe3a486e2609b53352f2d50a3148c4428066.nq.gz
    ├── 091e8698197584064974664474083a87d64f2908.nq.gz
    ├── 09bbb39a69cee7be3c028b01bcd2d7ba40663303.nq.gz
    ├── 0a1ae55b9718855772fb5197ca04875046a9323c.nq.gz
    ├── 0a270d14d3a4ad9eda62b68c2c01e9fdb710ef38.nq.gz
    ├── 0b50245f1ee819ede8dbdc177c10e06db48e8408.nq.gz
    ├── 0b64f641701c6bef5891c901d90b4108a3d87578.nq.gz
    ├── 0b6670800d2b012829bdf06b887f82ff3f554108.nq.gz
    ├── 0bfa91467d36228a5679c02e9441d3dcdeb4def6.nq.gz
    ├── 0c1e594993e01c3610c395608fe46ec6bde16214.nq.gz
    ├── 0c1f3b40021dbe886e5d338452da5ade7df5c76d.nq.gz
    ├── 0cafdb2d8a6b52f6ce5192d857d93bab376dc287.nq.gz
    ├── 0d66c55487cdb0f35fa86c4e10c33b0b9b0a5608.nq.gz
    ├── 0e2ccf0ba25c23e0ff292673a465a9a71e4eaef6.nq.gz
    ├── 0e2e4fe0c01a6c2510bdfc87bad2d3fb0ac934ce.nq.gz
    ├── 0e9bfffdf8af566c4bc13436361005c1e7b84dcb.nq.gz
    ├── 0f37ea0d9e105aa8d7a5e65c1f5b0450f018efe9.nq.gz
    ├── 0f4ddbfab10c11315a9de75f7dcc35cf7ddeae52.nq.gz
    ├── 105c6a1425651e130af0131d996fa029f8fb6bd7.nq.gz
    ├── 11cf06bb1fcd3ead072daad144c5bdc8bd2c37ba.nq.gz
    ├── 11e24914ffa48202a69c78af30155b065c95289d.nq.gz
    ├── 11faead5dac42bde6fa6adb3c896aa8efdfe1810.nq.gz
    ├── 129c574e391f9cada571712c604a2ce41157542c.nq.gz
    ├── 12fccaf3f2f9b54a4218f05cb339e4d345eb982d.nq.gz
    ├── 138adf1104e11708af0944553d008cedd81c225b.nq.gz
    ├── 13fbaa588fea9bf99857a5409136efeb486d19cb.nq.gz
    ├── 1499a97028eef11527829f13476152d00d8cb90b.nq.gz
    ├── 15292dcad97490aaa740cdec2d0aedb31e5662eb.nq.gz
    ├── 16b711ad5efce47d58efa808d5226dc7e52afd2b.nq.gz
    ├── 16bac9bfadcb99ebf16736cfa89bebc1dcc32e46.nq.gz
    ├── 16d62366068f3ac88e8bd61a7a9da2862547bcf8.nq.gz
    ├── 16ee8b2b4bc173ebb3079fc4d5e84738ba794d2d.nq.gz
    ├── 1808dc4f85b0bb77ac2fa469f17b5f903621f608.nq.gz
    ├── 18d981003d68d0546c4804ac2ff47dd97c6e7921.nq.gz
    ├── 19ec92278866073773d6c2b766d4fe37a9925929.nq.gz
    ├── 19fe468ac8103035ebb9dd87faa4f454f286de92.nq.gz
    ├── 1a198142874224a6766f321d9e0dfc97a01ecb43.nq.gz
    ├── 1a2dbabbd1cf5e4e37702b96865fd1af95a4f4d5.nq.gz
    ├── 1a32e6f2b2560917d2e3b36be397183dec848401.nq.gz
    ├── 1a3402466f4e03fa36c69260c1cf17cca893646d.nq.gz
    ├── 1a8b57c33bc9044cd649ba29b6aab989ee6d2e0c.nq.gz
    ├── 1af1049efbce650a979a3776d0e2e007fe0eb903.nq.gz
    ├── 1b8b1b08155ae339948c20d13f2f55d5a580a6bc.nq.gz
    ├── 1c1f3c7fa1bd1b6624e044759aea39d991202af9.nq.gz
    ├── 1c21353a056050cb048553a8813d318f4259631b.nq.gz
    ├── 1c25d9d350ef3aacc24fbe56943290c9a03add79.nq.gz
    ├── 1c427bb82ba737e433ebe4a87f7b70e366bb425f.nq.gz
    ├── 1cca56ddc561cf0068faecb847244a44e4db05a6.nq.gz
    ├── 1d317eb791515686c7294d8c0663f798df6fb71c.nq.gz
    ├── 1de871ce0fbea9ddbab7e315b05f864bc5f6fa53.nq.gz
    ├── 1e26b01a48c53e66ffa121cc7fb47d0a1e11cce2.nq.gz
    ├── 1edcf92c3317b90fedd187e2eaad101bd1c1efc5.nq.gz
    ├── 1efc513c91166243925d9f32cc2ae2d35de2f019.nq.gz
    ├── 1f50b1b05b530204bf32c6986fffc0aa8370ce54.nq.gz
    ├── 1f846beb6a0bccf8b545f5a67b74482015cc878b.nq.gz
    ├── 20291d09b9432b99a94f2241d2c2af76f4fde526.nq.gz
    ├── 202bbdbd0cd4192ebe6955a9576a9dd53c06f213.nq.gz
    ├── 2082c49cab9f735f69745c124436eb53cfda3ec7.nq.gz
    ├── 2098ac736acf8c7da6412afc41db978025a1ed83.nq.gz
    ├── 20bedc784d9b1337c5109088f5b32218ee69b3d1.nq.gz
    ├── 20ca3ca91e70d4496a6dd724842aa6946a440dce.nq.gz
    ├── 2108d1b315a73725115f22033954469a50718cb0.nq.gz
    ├── 222890da20ce8e1f3570706d4eb7394410c2cd09.nq.gz
    ├── 222be0ff359bd8c31ae82ea5cd4d4052e0733ead.nq.gz
    ├── 227425eef58ca6cf81d9ce9755c2df412f742ec4.nq.gz
    ├── 228aa2a0f9ae03a20dcca05310b0b265cd28a008.nq.gz
    ├── 22bcae27ab0e81cbd9899e5db185819f27c1f115.nq.gz
    ├── 22c14cbc08dc0178d7fb684eb1e7b408cf76f001.nq.gz
    ├── 22d0452f8d7e02ba33fa717d8a1792a76b050182.nq.gz
    ├── 22dd37216f8c64cdc497db01e19c08897ebc3f5c.nq.gz
    ├── 22fd59aa9c2f107864eda6a79f1bea7ac643710c.nq.gz
    ├── 23247e34178145af8249416e9c7756d87d680a99.nq.gz
    ├── 23683221f6005a9ce6a55e785e59409a649d7928.nq.gz
    ├── 23e841bf8749504030baca953e351dd9b7f146b0.nq.gz
    ├── 25084e154d674bbfb42d841de92255a8e16a63d4.nq.gz
    ├── 25214d6b13038149d5333c1bab16dc3fb6946396.nq.gz
    ├── 253920add718f75378017cfd1d7f78a4fd0b3cc7.nq.gz
    ├── 2543e1459892aa9dc474c3b4ccaf35c75b225fa3.nq.gz
    ├── 26b218a4dd78c54d5877383a2ab7703fcef8b0cd.nq.gz
    ├── 26c534486634ac5fbb0f1f1c60fed9353f5b69eb.nq.gz
    ├── 26f2c9ed199683bfae4427a75e49d094ef7c049e.nq.gz
    ├── 27716f4b6ab373fd686f1741333320588745ff91.nq.gz
    ├── 284aa92b2df6662c384cb6849a92d042254deacc.nq.gz
    ├── 28b9312e9c8825d8ee49a73215498525dd2ae199.nq.gz
    ├── 29a2382ed89f5d66f4b1869193c978fdfa458499.nq.gz
    ├── 29aaaaab334f9c32a4d9297bc690fcfd02986639.nq.gz
    ├── 2a50d9b8f45c672a59ebd81a430d8674682eb498.nq.gz
    ├── 2a7cc2a4a66a3de9a73a598671deef6c74e76e1b.nq.gz
    ├── 2acba6cbbdaebc6989f6a28bc68431d20f4e26ab.nq.gz
    ├── 2ae40af400fa9a201530fbb9760d67397be0812a.nq.gz
    ├── 2bf24bc3c80a94aa2ca56b26fd0e1495374d03ab.nq.gz
    ├── 2c2e10ffb2a779b7da15d3184e4e5531dedff8b4.nq.gz
    ├── 2c3c8459d5eb06b10643b11ab295a57846eb3792.nq.gz
    ├── 2c6ebbf90315a3fcfaaea650cd7d5c79f00abc77.nq.gz
    ├── 2c8e581dac17c3a4b07a600b9130cd1f4be8b277.nq.gz
    ├── 2c90690f4a55d709735b3df16e89c7d537e8bf22.nq.gz
    ├── 2cbe328ca8bebd421c9f29a3c217ef946a786034.nq.gz
    ├── 2d610d91dadefb270e0339ddbde32753ea2d2f32.nq.gz
    ├── 2d9eb3e661acf88dba27cf5eb0336f879d4593e6.nq.gz
    ├── 2da8b5b18f61abeb1c79534b9af597de1a0f42c4.nq.gz
    ├── 2dd9dafadde9fc11d58a98cc8c66480e50ed9ec2.nq.gz
    ├── 2ebd2961e1dee28468083356e3254670c170589f.nq.gz
    ├── 2ff17af5328cbc0995432560c86288f405cd5a46.nq.gz
    ├── 305149c87b115a7e6789979c224c71c53645d596.nq.gz
    ├── 3066c28ebd482a128f12656c966c246bfb8f0de9.nq.gz
    ├── 3066fb145925e944f10f5d8c56384ef89310290c.nq.gz
    ├── 30cd9c983f636b803e77215b6079edd1907a6c0b.nq.gz
    ├── 312249c53e1210f02ffb5254a577c90ebf334358.nq.gz
    ├── 3131b5e8c495ec763ccc822a43e19133eb5fd3ba.nq.gz
    ├── 315a83331ccf04f367654072ae58ee14f9c8e166.nq.gz
    ├── 31ab28ebdba2660ba5ec0a16b19361ad30a8a692.nq.gz
    ├── 320b1936d6f8897d6f324b6c4938dbe289fd466e.nq.gz
    ├── 3264cb1fd0ce43ca40cad4e8f0ca46e9cf1703db.nq.gz
    ├── 32ad1a32f897e11a3c1e05050f1c1f691b7a6936.nq.gz
    ├── 32d9542e692b2dbba3fc461655bc959083b8fa7e.nq.gz
    ├── 34dbdd074f187b71d5c37ed063958812e8c49425.nq.gz
    ├── 34fc3d4aa084462aab63490b5df13f7a29c39e4b.nq.gz
    ├── 367eb4ec128c6325794846385dd6763dca62b9a6.nq.gz
    ├── 3680775a390c54c902b4671242b2bd523c4abd0e.nq.gz
    ├── 36edbf4991214368f40920f1de21c4df52e0513f.nq.gz
    ├── 370633f2ec2b7cf35a37a283095229de337f46e4.nq.gz
    ├── 37db28b2badfdc4fd42ceaeb8aa301780d3b16f9.nq.gz
    ├── 387439c0433e8fa9f16163b1ad9629591639d09e.nq.gz
    ├── 38cde794e85abc92020a469d9bc69411cb4d5777.nq.gz
    ├── 390a25a0f8985767e8a9e39c43f6ad372befd1ca.nq.gz
    ├── 39482ceadbf931321368793c8cf7d305aea9cf16.nq.gz
    ├── 39c5d8f1bbaee7a6dcde7145f928b10b0f030616.nq.gz
    ├── 39f223f0398c836b9d109faf817526376fece7d2.nq.gz
    ├── 3a93545f6142a03cffb439a82bac80d41c97b098.nq.gz
    ├── 3a9374bb4381b03484a84707c6a0f76b23859cde.nq.gz
    ├── 3aaa038a9b4a3c459a1886c20627b73ea806286f.nq.gz
    ├── 3acb13046d4b2588a62269f3bd2cea51153e3a12.nq.gz
    ├── 3af97524bc2244cb59b98bac1658307a57af377f.nq.gz
    ├── 3b3520d70526038df68f835ce2a184e850cac66a.nq.gz
    ├── 3b9848594b61feaa5964bb89f5b82413d1b6d32a.nq.gz
    ├── 3bad0bbb02792af31e26adb3dfc6cec0375ae9a3.nq.gz
    ├── 3badd9264dc77f6f0b3f59cfeb1dd6e5da94b6f4.nq.gz
    ├── 3caa7434e20375252c9eb09f0553ac87fdaaa028.nq.gz
    ├── 3cda60fe0bc3effe5bf0ffceb5e01fb2adea4bf2.nq.gz
    ├── 3cf1e7f80d7128f6f1d3a5ea7b300dc0709ab0ee.nq.gz
    ├── 3d4e3e63f280c79044706fa5ac4e9c1c448fdefe.nq.gz
    ├── 3d66a701be175a8de7242d26b3c9831196a22e5c.nq.gz
    ├── 3d6f37f958a131b76ce80306718b77d78bc3f045.nq.gz
    ├── 3e77d72b904b14aebc99ff98d0b8acfbcedf7603.nq.gz
    ├── 3eb8443b54d1ae6cb727d21b23a5189715b90211.nq.gz
    ├── 3ecff65ec609902e0a57d0b6134d0c05b763a9e2.nq.gz
    ├── 3f47fdec65c38ac33de3c72f6ec71c8b745f1a56.nq.gz
    ├── 4005b4a90729c9fe1b811f7388bd8453998d2322.nq.gz
    ├── 403784b1db170f700d0621ed6440db550c47836c.nq.gz
    ├── 40bae605d028d3f522531711a1e28298b63ffbfc.nq.gz
    ├── 412b931299ebcc6afa4d677c514e1c2b21681545.nq.gz
    ├── 4146651c737971cc5a883b6750f2ded3051bc8ea.nq.gz
    ├── 41d32b205ee9b29dc7e25e087c77b3d789da0d83.nq.gz
    ├── 4201240a42725dc52e05e67859347c65459e7e8e.nq.gz
    ├── 4211155c455d05212e713ea5a988b3156915f0f3.nq.gz
    ├── 4217a9d24be00f95336d46cd1221e20cbe3b3930.nq.gz
    ├── 42743fb0c80fcdec19554b350d2b91149a56bdae.nq.gz
    ├── 42b9d65562d81f9ce1be56180c433de44d5e9b4f.nq.gz
    ├── 42bfdc5c68f1b42c9d36d216aa16ccb91cbeb169.nq.gz
    ├── 42efbe8de68332f99a55c379a3b4714dff261033.nq.gz
    ├── 4322470286d9f7edd1e1219e9f16fa963028040d.nq.gz
    ├── 4358389b3e50c2c7b025a3c097fecd80af5f6306.nq.gz
    ├── 43968762a8b0bdad084187ec190a1c3ba327ddda.nq.gz
    ├── 43a688604f69d894973d4a76f26450108aaf0413.nq.gz
    ├── 43b0df48ffe35c055e63362031088d18c24a2dbe.nq.gz
    ├── 44213104b523ffe909bd1ed7f624dd72457d0c3b.nq.gz
    ├── 442d9132ea32808ad980df4bd233b359f76341a7.nq.gz
    ├── 4437298f164eeb8e3914ec4e9f38e5f58e0b88f3.nq.gz
    ├── 4466d82291bfa908aff424bb66ae704289b97274.nq.gz
    ├── 44b917a6ec688b67cbe19e813eb1749e89f64ef0.nq.gz
    ├── 44ce8db6b8ed192c9ab1f7b3aab88d1fd49b4e43.nq.gz
    ├── 454ce118a6d2ac8ac98e355689f85133be3f7ed7.nq.gz
    ├── 4587f3798da415bb92f1b8e36e9c4e26d9a0caea.nq.gz
    ├── 45893a4499506a43323bf53d9552adec2a457261.nq.gz
    ├── 45bc2e7c43563ad5603f4c53cfee3064cce5e4c7.nq.gz
    ├── 461688405d7b162195aef9a9a6a5d3c293ac57a1.nq.gz
    ├── 46d0d96e997e0c388669b2f8c968f2809b4ac983.nq.gz
    ├── 46d23d955d4986d57582b073e751a5c2bfd4cb44.nq.gz
    └── 46d82f5e784954e10ea49487c865fc0c6921a313.nq.gz

8 directories, 200 files
```

| Directory | What it contains |
|-----------|-----------------|
| `blob/` | Per-file AST graphs, content-addressed by git blob SHA. Each file in the source repo gets its own graph. |
| `aggregate/ast/` | Combined AST graph per parsed commit. Merges all blob graphs for a snapshot of the entire codebase at that point. |
| `aggregate/lsp/` | Language Server Protocol enrichment: resolved symbols, definitions, references, and type information. |
| `aggregate/dataflow/` | Interprocedural data flow edges between functions and modules. |
| `aggregate/repolex/` | Combined graph (AST + LSP + dataflow) per commit. |
| `commit/` | Git commit metadata (author, date, message, parent links). |
| `branch/` | Branch metadata. |
| `tag/` | Tag metadata. |
| `filetree/` | File tree snapshots per commit (which files existed and their blob SHAs). |

## Source repository

[pytorch/vision](https://github.com/pytorch/vision)

---
*Parsed on 2026-04-10 by [repolex](https://repolex.ai)*
