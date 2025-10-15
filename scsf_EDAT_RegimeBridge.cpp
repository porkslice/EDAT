// scsf_EDAT_RegimeBridge.cpp — overlay Path B regime (fixed)
#include "sierrachart.h"
#include <vector>
#include <cstdlib>   // atoi

SCDLLName("EDAT Regime Bridge (fixed)")

static void ReadLastCSVLine(const SCString& path, SCString& out) {
    FILE* f = fopen(path.GetChars(), "rb"); if (!f) { out = ""; return; }
    fseek(f, 0, SEEK_END); long pos = ftell(f); long i = 1; int ch = 0;
    while (i <= pos) { fseek(f, -i, SEEK_END); ch = fgetc(f); if (ch=='\n' && i>1) break; ++i; }
    std::string line; while ((ch=fgetc(f))!=EOF && ch!='\n') line.push_back((char)ch);
    fclose(f);
    out = line.c_str();
}

SCSFExport scsf_EDAT_RegimeBridge(SCStudyInterfaceRef sc) {
    SCSubgraphRef RegBG = sc.Subgraph[0];
    SCInputRef InPath   = sc.Input[0];

    if (sc.SetDefaults) {
        sc.GraphName = "EDAT Regime Bridge (fixed)";
        sc.AutoLoop = 0;
        sc.GraphRegion = 0;                         // overlay in price pane
        sc.ScaleRangeType = SCALE_SAMEASREGION;

        RegBG.Name = "RegimeBG";
        RegBG.DrawStyle = DRAWSTYLE_BACKGROUND;
        RegBG.PrimaryColor = RGB(35,35,35);
        RegBG.DrawZeros = 0;

        InPath.Name = "CSV Path";
        InPath.SetString(sc.DataFilesFolder() + SCString("\\EDAT\\edat_regime_MNQ.csv"));
        return;
    }

    SCString path = InPath.GetString();
    SCString last; ReadLastCSVLine(path, last);
    if (last.IsEmpty()) return;

    // Tokenize CSV
    std::vector<char*> toks;
    last.Tokenize(",", toks);
    if (toks.size() < 8) return;       // need at least ts,state_id,...,tau

    int state = atoi(toks[1]);         // state_id is second column

    // pick color
    COLORREF bgc = RGB(35,35,35);      // Trans (default)
    if (state == 0) bgc = RGB(22,44,22);   // Range
    else if (state == 2) bgc = RGB(60,20,20);  // TrendUp
    else if (state == 3) bgc = RGB(20,20,60);  // TrendDn

    int idx = sc.ArraySize - 1;
    RegBG[idx] = sc.Close[idx];        // must set a value to paint
    RegBG.DataColor[idx] = bgc;
}

