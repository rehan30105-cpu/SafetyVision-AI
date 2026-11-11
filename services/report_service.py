import os
import io
import logging
import cv2
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from database.db import get_db
from services.recommendation_engine import generate_recommendations
from core.annotator import draw_person_annotations

logger = logging.getLogger("safetyvision.report")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'reports')
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'uploads')
os.makedirs(REPORTS_DIR, exist_ok=True)


def _infer_raw_detections(frame):
    """Run the existing YOLO inference once on a frame and return raw detections.

    Reuses the singleton model and the configured confidence/IoU thresholds.
    Returns [] on any failure so callers can degrade gracefully. This is the
    same inference the image/video routes already perform to annotate evidence;
    it is required here because raw per-box detections are not persisted with a
    session (only summarized person results are).
    """
    try:
        from core.model_manager import ModelManager
        from database.db import get_setting
        conf = float(get_setting('confidence_threshold', 0.35))
        iou = float(get_setting('iou_threshold', 0.45))
        dets, _ = ModelManager.get_instance().predict(frame, conf=conf, iou=iou)
        return dets or []
    except Exception as exc:
        logger.warning("[REPORT] inference for annotation failed: %s", type(exc).__name__)
        return []


def resolve_annotated_evidence(evidence_rel, person_results=None):
    """Return a report-ready evidence path that carries real bounding boxes.

    Draws EVERY valid PPE box present in the actual inference results, so a
    valid NO-Mask detection is shown even when anatomical association did not
    attach it to a person (association/counting are left untouched).

    Resolution order (never fabricates boxes):
      1. Locate the ORIGINAL raw frame (``evidence_*`` sibling when the stored
         path is already ``annotated_*``). If no raw frame exists, return the
         stored path unchanged.
      2. Run the existing inference once on the raw frame to obtain real
         per-box detections.
      3. Draw person boxes (from stored results) + all PPE/hazard boxes (from
         raw detections) via the existing annotator; save ``annotated_*`` (the
         ORIGINAL is never overwritten) and return it.
      4. On any failure, fall back to the stored path so reports never crash.
    """
    try:
        if not evidence_rel:
            return evidence_rel
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _abs(rel):
            return rel if os.path.isabs(rel) else os.path.join(root_dir, rel)

        base = os.path.basename(evidence_rel)
        # Work out the raw (un-annotated) frame and the annotated sibling name.
        if base.startswith('evidence_'):
            raw_rel = evidence_rel
            ann_rel = 'static/uploads/annotated_%s' % base[len('evidence_'):]
        elif base.startswith('annotated_'):
            ann_rel = evidence_rel
            raw_rel = 'static/uploads/evidence_%s' % base[len('annotated_'):]
        else:
            # Non-standard evidence name: keep as-is.
            return evidence_rel

        raw_abs = _abs(raw_rel)
        if not os.path.exists(raw_abs):
            # No original frame to re-annotate cleanly; use whatever we have.
            logger.info("[REPORT] no raw frame for %s; using stored %s", evidence_rel, evidence_rel)
            return evidence_rel

        frame = cv2.imread(raw_abs)
        if frame is None:
            logger.warning("[REPORT] could not read raw frame %s", raw_rel)
            return evidence_rel

        raw_dets = _infer_raw_detections(frame)
        persons = [p for p in (person_results or []) if isinstance(p, dict) and p.get('bbox')]
        # Nothing real to draw -> keep the original, do not fabricate/duplicate.
        if not persons and not raw_dets:
            logger.info("[REPORT] no detections/persons for %s; keeping original", evidence_rel)
            return evidence_rel
        annotated = draw_person_annotations(frame, person_results or [], raw_dets)
        cv2.imwrite(_abs(ann_rel), annotated)
        logger.info("[REPORT] annotated evidence (with raw boxes) | raw=%s | annotated=%s | persons=%d | raw_dets=%d",
                    raw_rel, ann_rel, len(persons), len(raw_dets))
        return ann_rel
    except Exception as exc:
        logger.warning("[REPORT] resolve_annotated_evidence failed for %s: %s", evidence_rel, type(exc).__name__)
        return evidence_rel


# ---------------------------------------------------------------------------
# Professional black / charcoal / orange report theme (matches the web UI)
# Shared by BOTH the single-session report and the consolidated overall
# report so every generated PDF speaks one design language.
# Palette mirrors static/css/style.css dark tokens:
#   page #0E0F13, surface #191A1F, card #212227, accent #FF941F,
#   text #F5F5F5, secondary #C6C6CA, muted #8C8C94.
# ---------------------------------------------------------------------------
C_BG = "#0E0F13"       # near-black page background
C_SURFACE = "#191A1F"  # dark charcoal panels
C_CARD = "#212227"     # slightly lighter charcoal (cards / tables)
C_ACCENT = "#FF941F"   # primary orange accent (website theme)
C_TEXT = "#F5F5F5"     # off-white primary text
C_TEXT2 = "#C6C6CA"    # muted-grey secondary text
C_BORDER = "#2A2D34"   # charcoal borders
C_SUCCESS = "#22C55E"  # compliant / success (status green)
C_WARNING = "#F5B942"  # warning (status amber)
C_DANGER = "#EF4444"   # PPE violation / critical (status red)

# Annotation colours used by core/annotator.py (BGR -> hex) for the legend.
_LEGEND = [
    ("#10B981", "Compliant PPE detected"),
    ("#EF4444", "Missing PPE (violation)"),
    ("#06B6D4", "Protective Mask"),
    ("#F97316", "High-risk person"),
    ("#F59E0B", "Caution / Safety Cone"),
    ("#EC4899", "Machinery hazard"),
]


def _esc(value):
    """Escape dynamic text for safe use inside ReportLab Paragraph markup."""
    s = "" if value is None else str(value)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num(value, suffix=""):
    """Format a numeric value, showing N/A when it is missing."""
    if value is None or value == "":
        return "N/A"
    try:
        if isinstance(value, float):
            txt = ("%.1f" % value).rstrip("0").rstrip(".")
        else:
            txt = str(value)
    except (TypeError, ValueError):
        return "N/A"
    return txt + suffix


def _status_for(sess):
    """Derive a safety status label + colour from real session data only."""
    viol = sess.get("violation_count") or 0
    crit = sess.get("critical_count") or 0
    comp = sess.get("overall_compliance")
    try:
        comp_f = float(comp)
    except (TypeError, ValueError):
        comp_f = None
    if crit and int(crit) > 0:
        return "CRITICAL", C_DANGER
    if viol and int(viol) > 0:
        return "UNSAFE", C_DANGER
    if comp_f is not None and comp_f < 70.0:
        return "CAUTION", C_WARNING
    return "SAFE", C_SUCCESS


def _session_page_decorator(session_uid):
    """Return an onPage callback that paints the dark background + footer."""
    def _draw(canvas, doc):
        w, h = A4
        # Page background
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(C_BG))
        canvas.rect(0, 0, w, h, fill=1, stroke=0)
        # Top accent bar
        canvas.setFillColor(colors.HexColor(C_ACCENT))
        canvas.rect(0, h - 4, w, 4, fill=1, stroke=0)
        # Footer divider
        canvas.setStrokeColor(colors.HexColor(C_BORDER))
        canvas.setLineWidth(0.5)
        canvas.line(30, 32, w - 30, 32)
        # Footer text
        canvas.setFillColor(colors.HexColor(C_TEXT2))
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(30, 23, "SafetyVision AI  |  Autonomous PPE Compliance Verification")
        canvas.drawCentredString(w / 2.0, 23, "Session: %s" % (session_uid or "N/A"))
        canvas.drawRightString(w - 30, 23, "Page %d" % doc.page)
        canvas.restoreState()
    return _draw


def generate_pdf_report(session_id=None, inspector="Chief Safety Inspector", date_filter=None, user_id=None, is_admin=False):
    """
    Generates a professional executive-ready PDF inspection report using ReportLab.
    Fetches real metrics from the SQLite database.
    """
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"SafetyVision_Report_{timestamp_str}_{os.urandom(4).hex()}.pdf"
    output_path = os.path.join(REPORTS_DIR, report_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Clean Enterprise Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#FFFFFF')
    )

    elements = []

    # 1. Header & Branding Banner
    elements.append(Paragraph("SAFETYVISION AI", title_style))
    elements.append(Paragraph("Intelligent PPE Compliance, Workplace Risk Monitoring & Automated Safety Inspection Report", subtitle_style))

    # Meta Info Table
    report_uid = f"SVR-{datetime.now().strftime('%Y%m%d')}-{timestamp_str[-4:]}"
    meta_data = [
        [
            Paragraph(f"<b>Report ID:</b> {report_uid}", body_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %b %Y %H:%M:%S')}", body_style),
            Paragraph(f"<b>Report Prepared For:</b> {_esc(inspector)}", body_style)
        ],
        [
            Paragraph("<b>Scope:</b> Account-owned records" if not is_admin else "<b>Scope:</b> System-wide records", body_style),
            Paragraph("<b>System Version:</b> SafetyVision v2.0 AI", body_style),
            Paragraph("<b>Assessment:</b> Recorded detections", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[2.5 * inch, 2.5 * inch, 2.2 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # 2. Database Metrics Gathering
    conn = get_db()
    cursor = conn.cursor()

    incident_scope = '1=1' if is_admin else 'user_id = ?'
    owner_params = [] if is_admin else [user_id]
    from services.session_service import SessionManager
    session_stats = SessionManager.get_instance().get_overall_stats(
        user_id=user_id, include_unowned=is_admin)
    cursor.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope}', owner_params)
    total_incidents = cursor.fetchone()[0]

    cursor.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND severity = "CRITICAL"', owner_params)
    critical_incidents = cursor.fetchone()[0]

    cursor.execute(f'SELECT COUNT(*) FROM incidents WHERE {incident_scope} AND status = "RESOLVED"', owner_params)
    resolved_incidents = cursor.fetchone()[0]

    cursor.execute(f'SELECT AVG(risk_score) FROM incidents WHERE {incident_scope}', owner_params)
    avg_risk_row = cursor.fetchone()[0]
    avg_risk = round(avg_risk_row, 1) if avg_risk_row is not None else 0.0

    # Violations breakdown
    cursor.execute(f'''
        SELECT violation_type, COUNT(*) as cnt, AVG(confidence) as avg_conf, AVG(duration_seconds) as avg_dur
        FROM incidents
        WHERE {incident_scope}
        GROUP BY violation_type
        ORDER BY cnt DESC
    ''', owner_params)
    violation_rows = cursor.fetchall()

    # Zone breakdown
    cursor.execute(f'''
        SELECT zone_name, COUNT(*) as cnt, AVG(risk_score) as avg_r
        FROM incidents
        WHERE {incident_scope} AND zone_name IS NOT NULL
        GROUP BY zone_name
        ORDER BY cnt DESC
    ''', owner_params)
    zone_rows = cursor.fetchall()

    # Evidence photos (up to 2 recent photos)
    cursor.execute(f'''
        SELECT evidence_image, violation_type, person_tracking_id, risk_score, timestamp
        FROM incidents
        WHERE {incident_scope} AND evidence_image IS NOT NULL AND evidence_image != ""
        ORDER BY id DESC LIMIT 2
    ''', owner_params)
    evidence_rows = cursor.fetchall()

    conn.close()

    # Use persisted session measurements; never infer compliance from incident
    # count or invent a baseline when no assessment exists.
    compliance_est = session_stats.get('overall_compliance')
    site_score = session_stats.get('safety_score')

    # 3. Executive KPI Dashboard Cards Table
    elements.append(Paragraph("Executive Safety Summary", section_heading))
    kpi_data = [
        [
            Paragraph(f"<b>Total Incidents</b><br/><font size=14 color='#0F172A'><b>{total_incidents}</b></font>", body_style),
            Paragraph(f"<b>Critical Hazards</b><br/><font size=14 color='#EF4444'><b>{critical_incidents}</b></font>", body_style),
            Paragraph(f"<b>Resolved Items</b><br/><font size=14 color='#10B981'><b>{resolved_incidents}</b></font>", body_style),
            Paragraph(f"<b>Average Risk</b><br/><font size=14 color='#F59E0B'><b>{avg_risk}/100</b></font>", body_style),
            Paragraph(f"<b>Safety Score</b><br/><font size=14 color='#2563EB'><b>{site_score if site_score is not None else 'N/A'}{'/100' if site_score is not None else ''}</b></font>", body_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[1.45 * inch] * 5)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 14))

    # 4. Violation Breakdown Table
    elements.append(Paragraph("PPE & Hazard Violation Breakdown", section_heading))
    viol_headers = [
        Paragraph("<b>Violation Category</b>", body_style),
        Paragraph("<b>Occurrences</b>", body_style),
        Paragraph("<b>Avg Confidence</b>", body_style),
        Paragraph("<b>Avg Duration (s)</b>", body_style),
        Paragraph("<b>Distribution</b>", body_style)
    ]
    viol_table_rows = [viol_headers]

    if violation_rows:
        for row in violation_rows:
            v_type, cnt, avg_c, avg_d = row
            pct = round((cnt / max(total_incidents, 1)) * 100, 1)
            viol_table_rows.append([
                Paragraph(str(v_type), body_style),
                Paragraph(str(cnt), body_style),
                Paragraph(f"{round(avg_c * 100, 1)}%" if avg_c is not None else "N/A", body_style),
                Paragraph(f"{round(avg_d or 0.0, 1)}s", body_style),
                Paragraph(f"{pct}%", body_style)
            ])
    else:
        viol_table_rows.append([
            Paragraph("No violations recorded in selected window", body_style),
            Paragraph("0", body_style), Paragraph("N/A", body_style), Paragraph("0.0s", body_style), Paragraph("0%", body_style)
        ])

    v_table = Table(viol_table_rows, colWidths=[2.6 * inch, 1.1 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ]))
    elements.append(v_table)
    elements.append(Spacer(1, 14))

    # 5. Zone Analysis
    elements.append(Paragraph("Safety Zone Compliance Analysis", section_heading))
    zone_table_data = [[
        Paragraph("<b>Zone Name</b>", body_style),
        Paragraph("<b>Incident Count</b>", body_style),
        Paragraph("<b>Average Risk</b>", body_style),
        Paragraph("<b>Risk Level</b>", body_style)
    ]]
    if zone_rows:
        for z in zone_rows:
            z_name, z_cnt, z_risk = z
            lvl = "HIGH" if (z_risk or 0) >= 60 else ("MEDIUM" if (z_risk or 0) >= 30 else "LOW")
            zone_table_data.append([
                Paragraph(str(z_name), body_style),
                Paragraph(str(z_cnt), body_style),
                Paragraph(f"{round(z_risk or 0, 1)}/100", body_style),
                Paragraph(lvl, body_style)
            ])
    else:
        zone_table_data.append([
            Paragraph("No zone-specific incident data", body_style), Paragraph("0", body_style), Paragraph("N/A", body_style), Paragraph("N/A", body_style)
        ])

    z_table = Table(zone_table_data, colWidths=[3.2 * inch, 1.3 * inch, 1.3 * inch, 1.4 * inch])
    z_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(z_table)
    elements.append(Spacer(1, 14))

    # 6. Evidence Frames (if available)
    if evidence_rows:
        elements.append(Paragraph("Automated Visual Incident Evidence", section_heading))
        ev_items = []
        for ev in evidence_rows:
            ev_img_rel, ev_viol, ev_pid, ev_risk, ev_ts = ev
            full_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ev_img_rel.replace('/', os.sep))
            if os.path.exists(full_img_path):
                try:
                    img_elem = Image(full_img_path, width=3.3 * inch, height=2.0 * inch)
                    caption = Paragraph(f"<b>Person #{ev_pid}</b> - {ev_viol}<br/>Risk: {ev_risk}/100 | {ev_ts}", body_style)
                    ev_items.append([img_elem, caption])
                except Exception:
                    pass
        if ev_items:
            ev_table = Table(ev_items, colWidths=[3.6 * inch, 3.6 * inch])
            ev_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(ev_table)
            elements.append(Spacer(1, 14))

    # 7. Recommendations
    elements.append(Paragraph("Automated Safety Directives & Recommendations", section_heading))
    recs = generate_recommendations(user_id=user_id, is_admin=is_admin)
    rec_table_rows = [[
        Paragraph("<b>Priority</b>", body_style),
        Paragraph("<b>Title & Observation</b>", body_style),
        Paragraph("<b>Recommended Corrective Action</b>", body_style)
    ]]
    for r in recs[:4]:
        rec_table_rows.append([
            Paragraph(f"<b>{r['priority']}</b>", body_style),
            Paragraph(f"<b>{r['title']}</b><br/>{r['observation']}", body_style),
            Paragraph(r['action'], body_style)
        ])
    r_table = Table(rec_table_rows, colWidths=[1.1 * inch, 3.1 * inch, 3.0 * inch])
    r_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(r_table)
    elements.append(Spacer(1, 18))

    # 8. Sign-off block
    elements.append(Paragraph(
        "<b>Report Note:</b> Metrics reflect the stored detection and incident records in the scope shown above. Model detections are automated assessments and require human review.",
        body_style
    ))

    # Build PDF
    doc.build(elements)

    return f"static/reports/{report_filename}"

def generate_session_pdf_report(session_uid, user_id=None, is_admin=False):
    """
    Generates a dedicated, standalone PDF report for an individual detection session.
    Contains ONLY that session's data, its evidence screenshot, and person-wise results.
    """
    from services.session_service import SessionManager
    mgr = SessionManager.get_instance()
    sess = mgr.get_session(session_uid, user_id=user_id, include_unowned=is_admin)
    if not sess:
        return None

    report_filename = f"SafetyVision_Session_{session_uid}_{os.urandom(4).hex()}.pdf"
    output_path = os.path.join(REPORTS_DIR, report_filename)

    # One-page A4 portrait. ~751pt usable height with these margins.
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=30, leftMargin=30, topMargin=26, bottomMargin=30
    )
    # Usable frame width inside the side margins (A4 width minus 60pt).
    avail_w = A4[0] - doc.leftMargin - doc.rightMargin

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, leading=16,
        textColor=colors.HexColor(C_TEXT), spaceAfter=1
    )
    accent_style = ParagraphStyle(
        'Accent', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=colors.HexColor(C_ACCENT), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=colors.HexColor(C_TEXT2), spaceAfter=6
    )
    section_heading = ParagraphStyle(
        'SecHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=9, leading=11,
        textColor=colors.HexColor(C_TEXT), spaceBefore=4, spaceAfter=3
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'], fontName='Helvetica', fontSize=7.2, leading=9.5,
        textColor=colors.HexColor(C_TEXT)
    )
    body2_style = ParagraphStyle(
        'DocBody2', parent=styles['Normal'], fontName='Helvetica', fontSize=7.2, leading=9.5,
        textColor=colors.HexColor(C_TEXT2)
    )
    kpi_label = ParagraphStyle(
        'KpiLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=6.4, leading=8,
        textColor=colors.HexColor(C_TEXT2), alignment=1
    )
    kpi_value = ParagraphStyle(
        'KpiValue', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, leading=14,
        textColor=colors.HexColor(C_TEXT), alignment=1
    )
    th_style = ParagraphStyle(
        'Th', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7, leading=8.5,
        textColor=colors.HexColor(C_BG)
    )
    td_style = ParagraphStyle(
        'Td', parent=styles['Normal'], fontName='Helvetica', fontSize=7, leading=9,
        textColor=colors.HexColor(C_TEXT)
    )

    elements = []

    # Derived values from the authoritative session row (real data only).
    total_p = sess.get('people_count', 0)
    comp_p = sess.get('compliant_count', 0)
    viol_p = sess.get('violation_count', 0)
    crit_p = sess.get('critical_count', 0)
    compliance_rate = sess.get('overall_compliance', None)
    safety_score = sess.get('safety_score', None)
    status_label, status_color = _status_for(sess)
    persons = sess.get('person_results', []) or []

    # ================= PAGE 1 — Safety Session Overview =================
    elements.append(Paragraph("SAFETYVISION AI", title_style))
    elements.append(Paragraph("Autonomous PPE Compliance Verification — Individual Session Report", subtitle_style))

    created = sess.get('created_at') or sess.get('start_time') or 'N/A'
    meta_data = [
        [
            Paragraph("<b>Session ID</b><br/>%s" % _esc(session_uid), body_style),
            Paragraph("<b>Detection Source</b><br/>%s" % _esc(sess.get('source_name') or 'Integrated Sensor'), body_style),
            Paragraph("<b>Capture Timestamp</b><br/>%s" % _esc(created), body_style),
        ],
        [
            Paragraph("<b>Session Type</b><br/>%s" % _esc(sess.get('session_type') or 'N/A'), body_style),
            Paragraph("<b>Duration</b><br/>%s" % _esc(_num(sess.get('duration_seconds'), "s")), body_style),
            Paragraph("<b>Completion Status</b><br/>%s" % _esc(sess.get('status', 'COMPLETED')), body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[2.5 * inch, 2.5 * inch, 2.35 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_CARD)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 5))

    # 3. Summary KPIs (themed cards)
    elements.append(Paragraph("Session Executive Summary", section_heading))

    def _kpi(label, value, value_color=C_TEXT):
        return [Paragraph(_esc(label), kpi_label),
                Paragraph("<font color='%s'>%s</font>" % (value_color, _esc(value)), kpi_value)]

    _cr = compliance_rate
    _cr_color = C_TEXT
    try:
        if _cr is not None:
            _cr_color = C_SUCCESS if float(_cr) >= 90 else (C_WARNING if float(_cr) >= 70 else C_DANGER)
    except (TypeError, ValueError):
        _cr_color = C_TEXT
    kpi_cells = [[
        _kpi("PEOPLE ANALYZED", _num(total_p)),
        _kpi("PPE VIOLATIONS", _num(viol_p), C_DANGER if (viol_p or 0) else C_SUCCESS),
        _kpi("OVERALL COMPLIANCE", _num(_cr, "%"), _cr_color),
        _kpi("SAFETY SCORE", _num(safety_score, "/100")),
        _kpi("CRITICAL HAZARDS", _num(crit_p), C_DANGER if (crit_p or 0) else C_SUCCESS),
    ]]
    kpi_table = Table(kpi_cells, colWidths=[1.47 * inch] * 5)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 5))

    # Safety status indicator
    _st_val = ParagraphStyle('st', parent=body_style, fontName='Helvetica-Bold', fontSize=9,
                             alignment=2, textColor=colors.HexColor(status_color))
    status_tbl = Table([[Paragraph("<b>OVERALL SAFETY STATUS</b>", body2_style),
                         Paragraph(status_label, _st_val)]],
                       colWidths=[4.85 * inch, 2.5 * inch])
    status_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor(status_color)),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(status_tbl)
    elements.append(Spacer(1, 5))

    # Brief session summary (built only from real data)
    if (total_p or 0) == 0:
        summary = "No personnel were detected during this session; no PPE compliance evaluation was performed."
    elif (viol_p or 0) == 0 and (crit_p or 0) == 0:
        summary = ("All %s detected personnel were fully compliant across mandatory PPE categories. "
                   "Overall compliance %s." % (_num(total_p), _num(compliance_rate, "%")))
    else:
        missing_cats = set()
        for p in persons:
            for m in (p.get('missing_items') or []):
                missing_cats.add(str(m))
        cat_txt = ", ".join(sorted(missing_cats)) if missing_cats else "PPE items"
        summary = ("%s personnel analyzed with %s PPE violation(s) and %s critical hazard(s). "
                   "Overall compliance %s. Missing items observed: %s." % (
                       _num(total_p), _num(viol_p), _num(crit_p), _num(compliance_rate, "%"), _esc(cat_txt)))
    elements.append(Paragraph("Session Summary", section_heading))
    elements.append(Paragraph(_esc(summary), body_style))

    # ================= Evidence (large centred figure, aspect preserved) ================
    # The evidence frame is the inspection centrepiece: it is rendered LARGE
    # (up to ~2.9in tall ≈ 40% taller than before) while keeping its original
    # aspect ratio, centred in a charcoal frame card. Transparent bounding-box
    # outlines and labels from the stored annotated file are preserved as-is
    # (never stretched, cropped, or fabricated). Margins/headings/tables around
    # it were compacted so the page still fits exactly one A4 sheet.
    ev_path = sess.get('evidence_image')
    if ev_path:
        ev_path = resolve_annotated_evidence(ev_path, sess.get('person_results'))
        logger.info("[REPORT] primary evidence for %s -> %s", session_uid, ev_path)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_ev_path = os.path.join(root_dir, ev_path) if not os.path.isabs(ev_path) else ev_path
        if os.path.exists(full_ev_path):
            try:
                img = cv2.imread(full_ev_path)
                iw, ih = (img.shape[1], img.shape[0]) if img is not None else (4, 3)
                max_w, max_h = avail_w, 208.0
                scale = min(max_w / float(iw), max_h / float(ih))
                im = Image(full_ev_path, width=iw * scale, height=ih * scale)
                im.hAlign = 'CENTER'
                frame_tbl = Table([[im]], colWidths=[avail_w])
                frame_tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_CARD)),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_ACCENT)),
                    ('LINEBELOW', (0, 0), (-1, 0), 2.0, colors.HexColor(C_ACCENT)),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(Paragraph("Primary Session Evidence", section_heading))
                elements.append(frame_tbl)
                elements.append(Spacer(1, 3))
            except Exception as e:
                logger.warning("[REPORT] evidence embed failed for %s: %s", session_uid, type(e).__name__)
    else:
        elements.append(Paragraph("No evidence image was captured for this session.", body2_style))
    elements.append(Spacer(1, 3))

    # Compact single-line annotation legend (bounded, no second figure).
    legend_style = ParagraphStyle('legend', parent=body2_style, fontSize=6.2, leading=8)
    legend_cells = []
    for hexc, label in _LEGEND:
        sw = Table([[""]], colWidths=[8], rowHeights=[8],
                   style=TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(hexc)),
                                     ('BOX', (0, 0), (-1, -1), 0.4, colors.HexColor(C_BG))]))
        legend_cells.append(sw)
        legend_cells.append(Paragraph(_esc(label), legend_style))
    legend_tbl = Table([legend_cells[:6], legend_cells[6:12], legend_cells[12:]],
                       colWidths=[10, 108, 10, 108, 10, 108])
    legend_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(legend_tbl)
    elements.append(Spacer(1, 3))


    # 5. Person-Wise PPE Results Table (themed, repeating header)
    elements.append(Paragraph("Individual Person PPE Analysis", section_heading))
    p_headers = [
        Paragraph("Person", th_style),
        Paragraph("Hardhat", th_style),
        Paragraph("Mask", th_style),
        Paragraph("Safety Vest", th_style),
        Paragraph("Compliance", th_style),
        Paragraph("Risk", th_style),
        Paragraph("Status / Missing Items", th_style),
    ]
    p_rows = [p_headers]

    if persons:
        for p in persons:
            hh_label = _esc(p.get('hardhat_label') or ('DETECTED' if p.get('hardhat') == 'PRESENT' else 'MISSING'))
            mk_label = _esc(p.get('mask_label') or ('DETECTED' if p.get('mask') == 'PRESENT' else 'MISSING'))
            vt_label = _esc(p.get('vest_label') or ('DETECTED' if p.get('vest') == 'PRESENT' else 'MISSING'))
            hh_text = "<font color='%s'><b>%s</b></font>" % (C_SUCCESS if p.get('hardhat') == 'PRESENT' else C_DANGER, hh_label)
            mk_text = "<font color='%s'><b>%s</b></font>" % (C_SUCCESS if p.get('mask') == 'PRESENT' else C_DANGER, mk_label)
            vt_text = "<font color='%s'><b>%s</b></font>" % (C_SUCCESS if p.get('vest') == 'PRESENT' else C_DANGER, vt_label)

            is_comp = (p.get('compliance', 0.0) >= 99.0)
            status_color = C_SUCCESS if is_comp else C_DANGER
            status_text = "<font color='%s'><b>%s</b></font>" % (status_color, _esc(p.get('status', 'PPE VIOLATION')))
            if not is_comp and p.get('missing_items'):
                status_text += "<br/><font color='%s'>Missing: %s</font>" % (C_TEXT2, _esc(", ".join(p.get('missing_items'))))

            p_rows.append([
                Paragraph("<b>%s</b>" % _esc(p.get('label', 'Person')), td_style),
                Paragraph(hh_text, td_style),
                Paragraph(mk_text, td_style),
                Paragraph(vt_text, td_style),
                Paragraph("<b>%s%%</b>" % _esc(_num(p.get('compliance'))), td_style),
                Paragraph("<b>%s</b>" % _esc(p.get('risk', 'LOW')), td_style),
                Paragraph(status_text, td_style),
            ])
    else:
        p_rows.append([
            Paragraph("No personnel detected during this session.", td_style),
            Paragraph("N/A", td_style), Paragraph("N/A", td_style), Paragraph("N/A", td_style),
            Paragraph("100%", td_style), Paragraph("LOW", td_style), Paragraph("COMPLIANT", td_style)
        ])

    p_table = Table(p_rows, colWidths=[0.9 * inch, 0.85 * inch, 0.85 * inch, 0.95 * inch, 0.85 * inch, 0.8 * inch, 1.15 * inch], repeatRows=1)
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(C_ACCENT)),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(C_SURFACE), colors.HexColor(C_CARD)]),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, colors.HexColor(C_BORDER)),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor(C_BORDER)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(p_table)
    elements.append(Spacer(1, 4))

    # 6. Corrective Actions (only if violations exist) — themed
    if (viol_p or 0) > 0:
        corrective_flowables = [
            Paragraph("Recommended Corrective Directives", section_heading),
        ]
        rec_data = [[
            Paragraph("Priority", th_style),
            Paragraph("Required Action", th_style),
            Paragraph("Compliance Standard", th_style),
        ]]
        if any(p.get('hardhat') == 'MISSING' for p in persons):
            rec_data.append([
                Paragraph("<font color='%s'><b>HIGH</b></font>" % C_DANGER, td_style),
                Paragraph("Issue immediate head protection gear (Hardhat) to unequipped worker before entering active zone.", td_style),
                Paragraph("OSHA 1926.100 / ANSI Z89.1", td_style),
            ])
        if any(p.get('vest') == 'MISSING' for p in persons):
            rec_data.append([
                Paragraph("<font color='%s'><b>MEDIUM</b></font>" % C_WARNING, td_style),
                Paragraph("Equip high-visibility reflective safety vest for optical personnel visibility.", td_style),
                Paragraph("ANSI/ISEA 107 Class 2", td_style),
            ])
        if any(p.get('mask') == 'MISSING' for p in persons):
            rec_data.append([
                Paragraph("<font color='%s'><b>MEDIUM</b></font>" % C_WARNING, td_style),
                Paragraph("Provide protective respiratory face covering for particulate hazard mitigation.", td_style),
                Paragraph("OSHA 1910.134 Respiratory Standard", td_style),
            ])
        r_table = Table(rec_data, colWidths=[0.9 * inch, 4.6 * inch, 1.85 * inch], repeatRows=1)
        r_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(C_ACCENT)),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(C_SURFACE), colors.HexColor(C_CARD)]),
            ('LINEBELOW', (0, 0), (-1, -1), 0.4, colors.HexColor(C_BORDER)),
            ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor(C_BORDER)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        corrective_flowables.append(r_table)
        corrective_flowables.append(Spacer(1, 4))
        elements.append(KeepTogether(corrective_flowables))
    else:
        elements.append(Paragraph(
            "<font color='%s'><b>NO RECORDED VIOLATIONS:</b> No violation events were recorded for this session.</font>" % C_SUCCESS,
            body_style
        ))
        elements.append(Spacer(1, 4))

    gen_ts = datetime.now().strftime('%d %b %Y %H:%M:%S')
    elements.append(Spacer(1, 3))
    elements.append(Paragraph(
        "<b>Report Note:</b> Computer-vision assessments are automated and may require human verification. "
        "Report generated %s." % _esc(gen_ts),
        body2_style
    ))

    doc.build(elements, onFirstPage=_session_page_decorator(session_uid),
              onLaterPages=_session_page_decorator(session_uid))
    return f"static/reports/{report_filename}"

def _overall_page_decorator():
    """Dark page chrome for the consolidated report: charcoal page, orange rule, footer."""
    def _decor(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(C_BG))
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(C_ACCENT))
        canvas.rect(0, doc.pagesize[1] - 7, doc.pagesize[0], 7, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(C_TEXT2))
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(doc.leftMargin, 26,
        "SAFETYVISION AI  •  Consolidated Workplace Safety Report  •  Confidential")
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 26,
                               "Page %d" % doc.page)
        canvas.restoreState()
    return _decor


def generate_overall_pdf_report(user_id=None, is_admin=False):
    """
    Generates an aggregated executive report combining all completed detection sessions.
    Uses the SAME dark/orange website design language as the single-session report
    (near-black page, charcoal panels, orange accents, off-white text).
    """
    from services.session_service import SessionManager
    mgr = SessionManager.get_instance()
    stats = mgr.get_overall_stats(user_id=user_id, include_unowned=is_admin)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"SafetyVision_Overall_Report_{timestamp_str}.pdf"
    output_path = os.path.join(REPORTS_DIR, report_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=30, leftMargin=30, topMargin=44, bottomMargin=40
    )
    avail_w = doc.width

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'OverallTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=17, leading=20,
        textColor=colors.HexColor(C_TEXT), spaceAfter=1
    )
    subtitle_style = ParagraphStyle(
        'OverallSub', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=12.5,
        textColor=colors.HexColor(C_ACCENT), spaceAfter=8
    )
    section_heading = ParagraphStyle(
        'OverallSec', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13,
        textColor=colors.HexColor(C_TEXT), borderPadding=(0, 0, 3, 6),
        spaceBefore=10, spaceAfter=5
    )
    body_style = ParagraphStyle(
        'OverallBody', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5,
        textColor=colors.HexColor(C_TEXT)
    )
    body2_style = ParagraphStyle(
        'OverallBody2', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=colors.HexColor(C_TEXT2)
    )
    hdr_style = ParagraphStyle(
        'OverallHdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, leading=9.5,
        textColor=colors.HexColor(C_BG)
    )
    kpi_label = ParagraphStyle(
        'OverallKpiLabel', parent=styles['Normal'], fontName='Helvetica', fontSize=6.5, leading=8,
        textColor=colors.HexColor(C_TEXT2)
    )
    kpi_value = ParagraphStyle(
        'OverallKpiValue', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=13,
        textColor=colors.HexColor(C_TEXT)
    )

    def _kpi(label, value, value_color=C_TEXT):
        return [Paragraph(_esc(label), kpi_label),
                Paragraph("<font color='%s'>%s</font>" % (value_color, _esc(value)), kpi_value)]

    def _num(value, suffix=""):
        if value is None or value == "":
            return "N/A"
        try:
            if isinstance(value, float):
                txt = ("%.1f" % value).rstrip("0").rstrip(".")
            else:
                txt = str(value)
        except (TypeError, ValueError):
            return "N/A"
        return txt + suffix

    elements = []

    # 1. Banner header (charcoal card, orange eyebrow + rule)
    scope_label = 'SYSTEM-WIDE' if is_admin else 'ACCOUNT-SCOPED'
    elements.append(Paragraph("<font color='%s'><b>CONSOLIDATED WORKPLACE AUDIT  •  %s SESSIONS</b></font>" % (C_ACCENT, scope_label), body2_style))
    elements.append(Paragraph("SAFETYVISION AI", title_style))
    elements.append(Paragraph("Consolidated Workplace Safety &amp; PPE Compliance Executive Audit Report", subtitle_style))
    head_tbl = Table([[
        Paragraph("<font color='%s'><b>Audit Date:</b> %s</font>" % (C_TEXT, _esc(datetime.now().strftime('%d %b %Y %H:%M'))), body2_style),
        Paragraph("<font color='%s'><b>Sessions Compiled:</b> %s</font>" % (C_TEXT, _esc(str(stats['total_sessions']))), body2_style),
        Paragraph("<font color='%s'><b>Scope:</b> %s</font>" % (C_TEXT, 'All user records' if is_admin else 'Owned account records'), body2_style),
    ]], colWidths=[avail_w * 0.36, avail_w * 0.30, avail_w * 0.34])
    head_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('LINEBELOW', (0, 0), (-1, 0), 2.0, colors.HexColor(C_ACCENT)),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(head_tbl)
    elements.append(Spacer(1, 4))

    # 2. Session-type split (image / video / live)
    elements.append(Paragraph("Session Mix", section_heading))
    type_cells = [[
        _kpi("IMAGE SESSIONS", _num(stats['image_sessions'])),
        _kpi("VIDEO SESSIONS", _num(stats['video_sessions'])),
        _kpi("LIVE SESSIONS", _num(stats['live_sessions'])),
    ]]
    type_tbl = Table(type_cells, colWidths=[avail_w / 3.0] * 3)
    type_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(type_tbl)
    elements.append(Spacer(1, 3))

    # 3. Overall KPIs (charcoal cards, status colours)
    elements.append(Paragraph("Cumulative Workplace Safety Metrics", section_heading))
    _oc = stats['overall_compliance']
    try:
        _oc_col = C_SUCCESS if float(_oc) >= 90 else (C_WARNING if float(_oc) >= 70 else C_DANGER)
    except (TypeError, ValueError):
        _oc_col = C_TEXT
    kpi_data = [[
        _kpi("TOTAL PEOPLE", _num(stats['total_people'])),
        _kpi("COMPLIANT", _num(stats['total_compliant']), C_SUCCESS),
        _kpi("VIOLATIONS", _num(stats['total_violations']), C_DANGER if stats['total_violations'] else C_SUCCESS),
        _kpi("CRITICAL", _num(stats['total_critical']), C_DANGER if stats['total_critical'] else C_SUCCESS),
        _kpi("PPE COMPLIANCE", _num(_oc, "%"), _oc_col),
        _kpi("SAFETY SCORE", _num(stats['safety_score'], "/100")),
    ]]
    kpi_table = Table(kpi_data, colWidths=[avail_w / 6.0] * 6)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(C_SURFACE)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('LINEBELOW', (0, 0), (-1, 0), 2.0, colors.HexColor(C_ACCENT)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 3))

    # 4. Equipment Breakdown (charcoal table, orange header band)
    elements.append(Paragraph("Equipment Infraction Breakdown", section_heading))
    gear_data = [
        [
            Paragraph("<b>Equipment Type</b>", hdr_style),
            Paragraph("<b>Observed Violations</b>", hdr_style),
            Paragraph("<b>Regulatory Mandate</b>", hdr_style)
        ],
        [
            Paragraph("<b>Hardhat (Head Protection)</b>", body_style),
            Paragraph("<font color='%s'><b>%s</b></font>" % (C_DANGER, _esc(str(stats['hardhat_violations']))), body_style),
            Paragraph("ANSI Z89.1 / OSHA 1926.100", body_style)
        ],
        [
            Paragraph("<b>Protective Mask (Respiratory)</b>", body_style),
            Paragraph("<font color='%s'><b>%s</b></font>" % (C_WARNING, _esc(str(stats['mask_violations']))), body_style),
            Paragraph("OSHA 1910.134 Respiratory Standard", body_style)
        ],
        [
            Paragraph("<b>Safety Vest (High-Visibility)</b>", body_style),
            Paragraph("<font color='%s'><b>%s</b></font>" % (C_ACCENT, _esc(str(stats['vest_violations']))), body_style),
            Paragraph("ANSI/ISEA 107 High-Visibility Class 2", body_style)
        ]
    ]
    g_table = Table(gear_data, colWidths=[avail_w * 0.36, avail_w * 0.25, avail_w * 0.39])
    g_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(C_ACCENT)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(C_BG)),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(C_CARD)),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(C_CARD), colors.HexColor(C_SURFACE)]),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(g_table)
    elements.append(Spacer(1, 4))

    # 5. Recent Sessions Table (charcoal table, orange header band)
    elements.append(Paragraph("Recent Completed Detection Sessions", section_heading))
    s_headers = [
        Paragraph("<b>Session ID</b>", hdr_style),
        Paragraph("<b>Type</b>", hdr_style),
        Paragraph("<b>Date</b>", hdr_style),
        Paragraph("<b>People</b>", hdr_style),
        Paragraph("<b>Compliant</b>", hdr_style),
        Paragraph("<b>Violations</b>", hdr_style),
        Paragraph("<b>Compliance %</b>", hdr_style)
    ]
    s_rows = [s_headers]
    for s in stats['recent_sessions']:
        s_rows.append([
            Paragraph("<b>%s</b>" % _esc(s['session_uid']), body2_style),
            Paragraph(_esc(s['session_type']), body2_style),
            Paragraph(_esc(s.get('created_at', '')[:10]), body2_style),
            Paragraph(_esc(str(s['people_count'])), body2_style),
            Paragraph("<font color='%s'>%s</font>" % (C_SUCCESS, _esc(str(s['compliant_count']))), body2_style),
            Paragraph("<font color='%s'>%s</font>" % (C_DANGER, _esc(str(s['violation_count']))), body2_style),
            Paragraph("<b>%s%%</b>" % _esc(str(s['overall_compliance'])), body2_style),
        ])
    if len(s_rows) == 1:
        s_rows.append([Paragraph("No completed sessions yet.", body2_style)] + [Paragraph("-", body2_style)] * 6)

    s_table = Table(s_rows, colWidths=[avail_w * 0.22, avail_w * 0.11, avail_w * 0.14, avail_w * 0.10, avail_w * 0.12, avail_w * 0.12, avail_w * 0.19])
    _s_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(C_ACCENT)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(C_BG)),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(C_CARD)),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor(C_BORDER)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(C_BORDER)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]
    if len(s_rows) > 2:
        _s_style.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(C_CARD), colors.HexColor(C_SURFACE)]))
    s_table.setStyle(TableStyle(_s_style))
    elements.append(s_table)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<font color='%s'><b>Report Note:</b> This report summarizes stored model assessments; it is not an independent safety certification.</font>" % C_TEXT2,
        body2_style
    ))

    doc.build(elements, onFirstPage=_overall_page_decorator(), onLaterPages=_overall_page_decorator())
    return f"static/reports/{report_filename}"

