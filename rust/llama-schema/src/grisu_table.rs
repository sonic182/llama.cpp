pub struct CachedPower {
    pub f: u64,
    pub e: i32,
    pub k: i32,
}

pub const CACHED_POWERS: [CachedPower; 79] = [
    CachedPower {
        f: 0xAB70FE17C79AC6CA,
        e: -1060,
        k: -300,
    },
    CachedPower {
        f: 0xFF77B1FCBEBCDC4F,
        e: -1034,
        k: -292,
    },
    CachedPower {
        f: 0xBE5691EF416BD60C,
        e: -1007,
        k: -284,
    },
    CachedPower {
        f: 0x8DD01FAD907FFC3C,
        e: -980,
        k: -276,
    },
    CachedPower {
        f: 0xD3515C2831559A83,
        e: -954,
        k: -268,
    },
    CachedPower {
        f: 0x9D71AC8FADA6C9B5,
        e: -927,
        k: -260,
    },
    CachedPower {
        f: 0xEA9C227723EE8BCB,
        e: -901,
        k: -252,
    },
    CachedPower {
        f: 0xAECC49914078536D,
        e: -874,
        k: -244,
    },
    CachedPower {
        f: 0x823C12795DB6CE57,
        e: -847,
        k: -236,
    },
    CachedPower {
        f: 0xC21094364DFB5637,
        e: -821,
        k: -228,
    },
    CachedPower {
        f: 0x9096EA6F3848984F,
        e: -794,
        k: -220,
    },
    CachedPower {
        f: 0xD77485CB25823AC7,
        e: -768,
        k: -212,
    },
    CachedPower {
        f: 0xA086CFCD97BF97F4,
        e: -741,
        k: -204,
    },
    CachedPower {
        f: 0xEF340A98172AACE5,
        e: -715,
        k: -196,
    },
    CachedPower {
        f: 0xB23867FB2A35B28E,
        e: -688,
        k: -188,
    },
    CachedPower {
        f: 0x84C8D4DFD2C63F3B,
        e: -661,
        k: -180,
    },
    CachedPower {
        f: 0xC5DD44271AD3CDBA,
        e: -635,
        k: -172,
    },
    CachedPower {
        f: 0x936B9FCEBB25C996,
        e: -608,
        k: -164,
    },
    CachedPower {
        f: 0xDBAC6C247D62A584,
        e: -582,
        k: -156,
    },
    CachedPower {
        f: 0xA3AB66580D5FDAF6,
        e: -555,
        k: -148,
    },
    CachedPower {
        f: 0xF3E2F893DEC3F126,
        e: -529,
        k: -140,
    },
    CachedPower {
        f: 0xB5B5ADA8AAFF80B8,
        e: -502,
        k: -132,
    },
    CachedPower {
        f: 0x87625F056C7C4A8B,
        e: -475,
        k: -124,
    },
    CachedPower {
        f: 0xC9BCFF6034C13053,
        e: -449,
        k: -116,
    },
    CachedPower {
        f: 0x964E858C91BA2655,
        e: -422,
        k: -108,
    },
    CachedPower {
        f: 0xDFF9772470297EBD,
        e: -396,
        k: -100,
    },
    CachedPower {
        f: 0xA6DFBD9FB8E5B88F,
        e: -369,
        k: -92,
    },
    CachedPower {
        f: 0xF8A95FCF88747D94,
        e: -343,
        k: -84,
    },
    CachedPower {
        f: 0xB94470938FA89BCF,
        e: -316,
        k: -76,
    },
    CachedPower {
        f: 0x8A08F0F8BF0F156B,
        e: -289,
        k: -68,
    },
    CachedPower {
        f: 0xCDB02555653131B6,
        e: -263,
        k: -60,
    },
    CachedPower {
        f: 0x993FE2C6D07B7FAC,
        e: -236,
        k: -52,
    },
    CachedPower {
        f: 0xE45C10C42A2B3B06,
        e: -210,
        k: -44,
    },
    CachedPower {
        f: 0xAA242499697392D3,
        e: -183,
        k: -36,
    },
    CachedPower {
        f: 0xFD87B5F28300CA0E,
        e: -157,
        k: -28,
    },
    CachedPower {
        f: 0xBCE5086492111AEB,
        e: -130,
        k: -20,
    },
    CachedPower {
        f: 0x8CBCCC096F5088CC,
        e: -103,
        k: -12,
    },
    CachedPower {
        f: 0xD1B71758E219652C,
        e: -77,
        k: -4,
    },
    CachedPower {
        f: 0x9C40000000000000,
        e: -50,
        k: 4,
    },
    CachedPower {
        f: 0xE8D4A51000000000,
        e: -24,
        k: 12,
    },
    CachedPower {
        f: 0xAD78EBC5AC620000,
        e: 3,
        k: 20,
    },
    CachedPower {
        f: 0x813F3978F8940984,
        e: 30,
        k: 28,
    },
    CachedPower {
        f: 0xC097CE7BC90715B3,
        e: 56,
        k: 36,
    },
    CachedPower {
        f: 0x8F7E32CE7BEA5C70,
        e: 83,
        k: 44,
    },
    CachedPower {
        f: 0xD5D238A4ABE98068,
        e: 109,
        k: 52,
    },
    CachedPower {
        f: 0x9F4F2726179A2245,
        e: 136,
        k: 60,
    },
    CachedPower {
        f: 0xED63A231D4C4FB27,
        e: 162,
        k: 68,
    },
    CachedPower {
        f: 0xB0DE65388CC8ADA8,
        e: 189,
        k: 76,
    },
    CachedPower {
        f: 0x83C7088E1AAB65DB,
        e: 216,
        k: 84,
    },
    CachedPower {
        f: 0xC45D1DF942711D9A,
        e: 242,
        k: 92,
    },
    CachedPower {
        f: 0x924D692CA61BE758,
        e: 269,
        k: 100,
    },
    CachedPower {
        f: 0xDA01EE641A708DEA,
        e: 295,
        k: 108,
    },
    CachedPower {
        f: 0xA26DA3999AEF774A,
        e: 322,
        k: 116,
    },
    CachedPower {
        f: 0xF209787BB47D6B85,
        e: 348,
        k: 124,
    },
    CachedPower {
        f: 0xB454E4A179DD1877,
        e: 375,
        k: 132,
    },
    CachedPower {
        f: 0x865B86925B9BC5C2,
        e: 402,
        k: 140,
    },
    CachedPower {
        f: 0xC83553C5C8965D3D,
        e: 428,
        k: 148,
    },
    CachedPower {
        f: 0x952AB45CFA97A0B3,
        e: 455,
        k: 156,
    },
    CachedPower {
        f: 0xDE469FBD99A05FE3,
        e: 481,
        k: 164,
    },
    CachedPower {
        f: 0xA59BC234DB398C25,
        e: 508,
        k: 172,
    },
    CachedPower {
        f: 0xF6C69A72A3989F5C,
        e: 534,
        k: 180,
    },
    CachedPower {
        f: 0xB7DCBF5354E9BECE,
        e: 561,
        k: 188,
    },
    CachedPower {
        f: 0x88FCF317F22241E2,
        e: 588,
        k: 196,
    },
    CachedPower {
        f: 0xCC20CE9BD35C78A5,
        e: 614,
        k: 204,
    },
    CachedPower {
        f: 0x98165AF37B2153DF,
        e: 641,
        k: 212,
    },
    CachedPower {
        f: 0xE2A0B5DC971F303A,
        e: 667,
        k: 220,
    },
    CachedPower {
        f: 0xA8D9D1535CE3B396,
        e: 694,
        k: 228,
    },
    CachedPower {
        f: 0xFB9B7CD9A4A7443C,
        e: 720,
        k: 236,
    },
    CachedPower {
        f: 0xBB764C4CA7A44410,
        e: 747,
        k: 244,
    },
    CachedPower {
        f: 0x8BAB8EEFB6409C1A,
        e: 774,
        k: 252,
    },
    CachedPower {
        f: 0xD01FEF10A657842C,
        e: 800,
        k: 260,
    },
    CachedPower {
        f: 0x9B10A4E5E9913129,
        e: 827,
        k: 268,
    },
    CachedPower {
        f: 0xE7109BFBA19C0C9D,
        e: 853,
        k: 276,
    },
    CachedPower {
        f: 0xAC2820D9623BF429,
        e: 880,
        k: 284,
    },
    CachedPower {
        f: 0x80444B5E7AA7CF85,
        e: 907,
        k: 292,
    },
    CachedPower {
        f: 0xBF21E44003ACDD2D,
        e: 933,
        k: 300,
    },
    CachedPower {
        f: 0x8E679C2F5E44FF8F,
        e: 960,
        k: 308,
    },
    CachedPower {
        f: 0xD433179D9C8CB841,
        e: 986,
        k: 316,
    },
    CachedPower {
        f: 0x9E19DB92B4E31BA9,
        e: 1013,
        k: 324,
    },
];
