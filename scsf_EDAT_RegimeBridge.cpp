// scsf_EDAT_RegimeBridge.cpp — v3: tint + arrows via subgraphs + HUD + optional η/ζ pane (compat-safe)
#include "sierrachart.h"
#include <vector>
#include <cstdlib>

SCDLLName("EDAT Regime Bridge v3")

// Read the last non-empty line from CSV
static void ReadLastCSVLine(const SCString& path, SCString& out)
{
    FILE* f = fopen(path.GetChars(), "rb");
    if (!f) { out = ""; return; }
    fseek(f, 0, SEEK_END);
    long endpos = ftell(f);
    if (endpos <= 0) { fclose(f); out = ""; return; }

    long i = 1; int ch = 0;
    while (i <= endpos) {
        fseek(f, -i, SEEK_END);
        ch = fgetc(f);
        if (ch == '\n' && i > 1) break;
        ++i;
    }
    std::string line;
    while ((ch = fgetc(f)) != EOF && ch != '\n') line.push_back((char)ch);
    fclose(f);
    out = line.c_str();
}

SCSFExport scsf_EDAT_RegimeBridge(SCStudyInterfaceRef sc)
{
    // Subgraphs
    SCSubgraphRef RegBG     = sc.Subgraph[0]; // background
    SCSubgraphRef EtaLine   = sc.Subgraph[1]; // metrics pane
    SCSubgraphRef ZetaLine  = sc.Subgraph[2]; // metrics pane
    SCSubgraphRef UpSig     = sc.Subgraph[3]; // arrow up on state change
    SCSubgraphRef DnSig     = sc.Subgraph[4]; // arrow down on state change
    SCSubgraphRef DotSig    = sc.Subgraph[5]; // dot on non-trend change

    // Inputs
    SCInputRef InPath       = sc.Input[0];
    SCInputRef ShowHUD      = sc.Input[1];
    SCInputRef ShowMetrics  = sc.Input[2];
    SCInputRef MetricsReg   = sc.Input[3];

    // Persistents
    int& PrevState = sc.GetPersistentInt(1);

    if (sc.SetDefaults)
    {
        sc.GraphName = "EDAT Regime Bridge v3";
        sc.AutoLoop = 0;               // manual updates
        sc.GraphRegion = 0;            // overlay
        sc.ScaleRangeType = SCALE_SAMEASREGION;

        RegBG.Name = "RegimeBG"; RegBG.DrawStyle = DRAWSTYLE_BACKGROUND; RegBG.PrimaryColor = RGB(35,35,35); RegBG.DrawZeros = 0;

        EtaLine.Name = "eta";  EtaLine.DrawStyle = DRAWSTYLE_LINE; EtaLine.PrimaryColor = RGB(0,170,0);   EtaLine.LineWidth = 2; EtaLine.DrawZeros = 0;
        ZetaLine.Name= "zeta"; ZetaLine.DrawStyle= DRAWSTYLE_LINE; ZetaLine.PrimaryColor= RGB(220,80,40); ZetaLine.LineWidth= 2; ZetaLine.DrawZeros = 0;

        UpSig.Name = "state_up";   UpSig.DrawStyle = DRAWSTYLE_ARROWDOWN; // tip points to bar (for long, arrow from above)
        UpSig.PrimaryColor = RGB(220,80,80); UpSig.LineWidth = 2; UpSig.DrawZeros = 0;
        DnSig.Name = "state_dn";   DnSig.DrawStyle = DRAWSTYLE_ARROWUP;   // tip points to bar (for short, arrow from below)
        DnSig.PrimaryColor = RGB(80,80,220); DnSig.LineWidth = 2; DnSig.DrawZeros = 0;
        DotSig.Name = "state_dot"; DotSig.DrawStyle = DRAWSTYLE_POINT;    // neutral change dot
        DotSig.PrimaryColor = RGB(160,160,160); DotSig.LineWidth = 3; DotSig.DrawZeros = 0;

        InPath.Name = "CSV Path";
        InPath.SetString(sc.DataFilesFolder() + SCString("\\EDAT\\edat_regime_MNQ.csv"));

        ShowHUD.Name = "Show HUD"; ShowHUD.SetYesNo(1);
        ShowMetrics.Name = "Show Metrics Pane"; ShowMetrics.SetYesNo(1);
        MetricsReg.Name = "Metrics Region Index"; MetricsReg.SetInt(2);

        PrevState = -999;
        return;
    }

    // Read last CSV row
    SCString path = InPath.GetString();
    SCString last; ReadLastCSVLine(path, last);
    if (last.IsEmpty()) return;

    // Tokenize: ts,state_id,state_name,p0,p1,p2,p3,tau,eps,zeta,phi,omega,eta,S
    std::vector<char*> tok;
    last.Tokenize(",", tok);
    if (tok.size() < 14) return;

    const int state = std::atoi(tok[1]);
    const int tau   = std::atoi(tok[7]);
    const double zeta = std::atof(tok[9]);
    const double eta  = std::atof(tok[12]);

    const int idx = sc.ArraySize - 1;
    if (idx < 0) return;

    // Background color by state
    COLORREF bgc = RGB(35,35,35);          // Trans default
    if (state == 0)      bgc = RGB(22,44,22);  // Range
    else if (state == 2) bgc = RGB(60,20,20);  // TrendUp
    else if (state == 3) bgc = RGB(20,20,60);  // TrendDn

    RegBG[idx] = sc.Close[idx];            // must set a value
    RegBG.DataColor[idx] = bgc;

    // Clear signals this bar
    UpSig[idx] = 0; DnSig[idx] = 0; DotSig[idx] = 0;

    // State-change markers via subgraphs (compatable with older SDKs)
    if (PrevState != -999 && PrevState != state)
    {
        // place a marker slightly above/below bar to avoid overlapping
        const float hi = sc.High[idx];
        const float lo = sc.Low[idx];
        const float pad = (hi - lo) * 0.2f + (float)sc.TickSize * 2;

        if (state == 2) {           // TrendUp
            UpSig[idx] = hi + pad;  // red arrow pointing down to the bar
        } else if (state == 3) {    // TrendDn
            DnSig[idx] = lo - pad;  // blue arrow pointing up to the bar
        } else {                    // any other change
            DotSig[idx] = sc.Close[idx];
        }
    }
    PrevState = state;

    // HUD (top-right-ish): anchor at last bar, a bit above high
    if (ShowHUD.GetYesNo())
    {
        const double p0 = std::atof(tok[3]);
        const double p1 = std::atof(tok[4]);
        const double p2 = std::atof(tok[5]);
        const double p3 = std::atof(tok[6]);
        double pmax = p0; if (p1>pmax) pmax=p1; if (p2>pmax) pmax=p2; if (p3>pmax) pmax=p3;

        s_UseTool hud; hud.Clear();
        hud.ChartNumber = sc.ChartNumber; hud.Region = 0;
        hud.AddMethod = UTAM_ADD_OR_ADJUST; hud.AllowSaveToChartbook = 0;
        hud.DrawingType = DRAWING_TEXT; hud.Color = RGB(235,235,235);
        hud.FontBold = 1; hud.FontSize = 12;
        hud.Text.Format("EDAT %d | τ=%d | p=%.2f | η=%.2f | ζ=%.2f", state, tau, pmax, eta, zeta);
        // position near last bar high + offset
        double y = sc.High[idx] + (sc.High[idx] - sc.Low[idx]) * 0.6 + sc.TickSize * 4;
        hud.BeginIndex = idx; hud.BeginValue = y;
        sc.UseTool(hud);
    }

    // Optional η/ζ pane
    if (ShowMetrics.GetYesNo())
    {
        sc.GraphRegion = MetricsReg.GetInt();
        EtaLine[idx]  = (float)eta;
        ZetaLine[idx] = (float)zeta;
    }
}
