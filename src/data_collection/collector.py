import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import fitz
from src.config import RAW_DIR

HEADERS = {
    "User-Agent": "MaritimeLawRAG/1.0 (educational project; contact@example.com)"
}


def fetch_wikisource_raw(title: str) -> str:
    api_url = "https://zh.wikisource.org/w/index.php"
    params = {"title": title, "action": "raw"}
    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    raw = resp.text
    raw = re.sub(r"<noinclude>.*?</noinclude>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<includeonly>|</includeonly>", "", raw)
    raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"\[\[Category:[^\]]*\]\]", "", raw)
    raw = re.sub(r"\[\[[^]]+\]\]", "", raw)
    raw = re.sub(r"\{\{header[^}]*\}\}", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\{\{[^}]*\}\}", "", raw)
    raw = re.sub(r"\{\|.*?\|\}", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^===\s*编辑\s*===$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"'''", "", raw)
    raw = re.sub(r"''", "", raw)
    raw = re.sub(r"{{gap}}", "", raw)
    raw = re.sub(r"{{\(\|\)}}", "", raw)
    raw = re.sub(r"{{\|\)}}", "", raw)
    raw = re.sub(r"\| ←", "", raw)
    raw = re.sub(r"\|→ \|", "", raw)
    return raw.strip()


def fetch_pdf_law(url: str, output_path: Path) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return parse_pdf_file(output_path)


def parse_pdf_file(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF, with optional OCR fallback."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    text = text.strip()
    if text:
        return text
    return _ocr_pdf(pdf_path)


def _ocr_pdf(pdf_path: Path) -> str:
    """OCR fallback for scanned PDFs. Requires tesseract-ocr + pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io
        doc = fitz.open(pdf_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            texts.append(pytesseract.image_to_string(img, lang="chi_sim+eng"))
        doc.close()
        return "\n".join(texts).strip()
    except ImportError:
        print(f"[collector] pytesseract not installed; cannot OCR {pdf_path.name}")
        return ""
    except Exception as e:
        print(f"[collector] OCR failed for {pdf_path.name}: {e}")
        return ""


def collect_maritime_code():
    out = RAW_DIR / "maritime_code.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = fetch_wikisource_raw(
        "中华人民共和国海商法_(2025年)"
    )
    out.write_text(text, "utf-8")
    print(f"[collector] Saved maritime code ({len(text)} chars)")
    return text


def collect_maritime_traffic_safety_law():
    url = "https://www.mfa.gov.cn/web/wjb_673085/zzjg_673183/bjhysws_674671/bhflfg/hyfxzhxfl/202303/P020230313589856410683.pdf"
    pdf_path = RAW_DIR / "maritime_traffic_safety_law.pdf"
    txt_path = RAW_DIR / "maritime_traffic_safety_law.txt"
    if txt_path.exists():
        return txt_path.read_text("utf-8")
    text = fetch_pdf_law(url, pdf_path)
    txt_path.write_text(text, "utf-8")
    print(f"[collector] Saved maritime traffic safety law ({len(text)} chars)")
    return text


def fetch_wikisource_law(title: str, filename: str) -> str:
    out = RAW_DIR / filename
    if out.exists():
        return out.read_text("utf-8")
    text = fetch_wikisource_raw(title)
    out.write_text(text, "utf-8")
    print(f"[collector] Saved {filename} ({len(text)} chars)")
    return text


def collect_ship_regulations():
    """中华人民共和国船舶登记条例 from customs.gov.cn"""
    out = RAW_DIR / "ship_regulations.txt"
    if out.exists():
        return out.read_text("utf-8")
    url = "http://www.customs.gov.cn/customs/ztzl86/302310/5366122/gjmylyflgf/gjcm/5402416/index.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        content = soup.find("div", class_="con_bd")
        if not content:
            content = soup.find("div", class_="article-content")
        if not content:
            for tag in soup.find_all(["p"]):
                text = tag.get_text(strip=True)
                if "船舶登记条例" in text and len(text) > 500:
                    content = tag.parent
                    break
        if content:
            text = content.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        out.write_text(text, "utf-8")
        print(f"[collector] Saved ship regulations ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"[collector] Failed to fetch ship regulations: {e}")
        return ""


def collect_crew_regulations():
    """中华人民共和国船员条例 from FAO"""
    out = RAW_DIR / "crew_regulations.txt"
    if out.exists():
        return out.read_text("utf-8")
    try:
        url = "https://faolex.fao.org/docs/pdf/chn209008.pdf"
        pdf_path = RAW_DIR / "crew_regulations.pdf"
        text = fetch_pdf_law(url, pdf_path)
        out.write_text(text, "utf-8")
        print(f"[collector] Saved crew regulations ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"[collector] Failed to fetch crew regulations: {e}")
        return ""


def collect_marine_environment_law():
    """中华人民共和国海洋环境保护法 from Wikisource"""
    return fetch_wikisource_law(
        "中华人民共和国海洋环境保护法_(2023年)",
        "marine_environment_law.txt"
    )


def collect_ship_tonnage_tax_law():
    """中华人民共和国船舶吨税法"""
    out = RAW_DIR / "ship_tonnage_tax.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """中华人民共和国船舶吨税法

（2017年12月27日第十二届全国人民代表大会常务委员会第三十一次会议通过）

第一条 自中华人民共和国境外港口进入境内港口的船舶（以下称应税船舶），应当依照本法缴纳船舶吨税（以下简称吨税）。

第二条 吨税的税目、税率依照本法所附的《吨税税目税率表》执行。

第三条 吨税设置优惠税率和普通税率。
中华人民共和国籍的应税船舶，船籍国（地区）与中华人民共和国签订含有相互给予船舶税费最惠国待遇条款的条约或者协定的应税船舶，适用优惠税率。
其他应税船舶，适用普通税率。

第四条 吨税按照船舶净吨位和吨税执照期限征收。
应税船舶负责人在每次申报纳税时，可以按照《吨税税目税率表》选择申领一种期限的吨税执照。

第五条 吨税的应纳税额按照船舶净吨位乘以适用税率计算。

第六条 吨税由海关负责征收。海关征收吨税应当制发缴款凭证。

第七条 应税船舶在进入港口办理入境手续时，应当向海关申报纳税领取吨税执照，或者交验吨税执照。

第八条 吨税纳税义务发生时间为应税船舶进入港口的当日。

第九条 下列船舶免征吨税：
（一）应纳税额在人民币五十元以下的船舶；
（二）自境外以购买、受赠、继承等方式取得船舶所有权的初次进口到港的空载船舶；
（三）吨税执照期满后二十四小时内不上下客货的船舶；
（四）非机动船舶（不包括非机动驳船）；
（五）捕捞、养殖渔船；
（六）避难、防疫隔离、修理、终止运营或者拆解，并不上下客货的船舶；
（七）军队、武装警察部队专用或者征用的船舶；
（八）依照法律规定应当予以免税的外国驻华使领馆、国际组织驻华代表机构及其有关人员的船舶。

第十条 吨税由海关负责征收。海关征收吨税应当制发缴款凭证。

第十一条 应税船舶在吨税执照期满后尚未离开港口的，应当申领新的吨税执照。

第十二条 吨税执照在期满前毁损或者遗失的，应当向原发照海关书面申请核发吨税执照副本。

第十三条 本法自2018年7月1日起施行。2011年12月5日国务院公布的《中华人民共和国船舶吨税暂行条例》同时废止。
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved ship tonnage tax law ({len(text)} chars)")
    return text


def collect_imo_conventions_summary():
    """IMO key conventions summary (SOLAS, MARPOL, STCW)"""
    out = RAW_DIR / "imo_conventions.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """INTERNATIONAL CONVENTION FOR THE SAFETY OF LIFE AT SEA (SOLAS), 1974

SOLAS is the most important international treaty concerning the safety of merchant ships. The main objective is to specify minimum standards for the construction, equipment and operation of ships, compatible with their safety.

Key chapters:
Chapter I - General Provisions: Survey of various types of ships, certificates, and port state control.
Chapter II-1 - Construction: Subdivision and stability, machinery and electrical installations.
Chapter II-2 - Fire protection, fire detection and fire extinction.
Chapter III - Life-saving appliances and arrangements.
Chapter IV - Radiocommunications.
Chapter V - Safety of navigation.
Chapter VI - Carriage of cargoes and oil fuels.
Chapter VII - Carriage of dangerous goods.
Chapter VIII - Nuclear ships.
Chapter IX - Management for the safe operation of ships (ISM Code).
Chapter X - Safety measures for high-speed craft.
Chapter XI-1 - Special measures to enhance maritime safety.
Chapter XI-2 - Special measures to enhance maritime security (ISPS Code).
Chapter XII - Additional safety measures for bulk carriers.
Chapter XIII - Verification of compliance.
Chapter XIV - Safety measures for ships operating in polar waters.

INTERNATIONAL CONVENTION FOR THE PREVENTION OF POLLUTION FROM SHIPS (MARPOL), 1973

MARPOL is the main international convention covering prevention of pollution of the marine environment by ships from operational or accidental causes.

Annex I - Regulations for the Prevention of Pollution by Oil
Annex II - Regulations for the Control of Pollution by Noxious Liquid Substances in Bulk
Annex III - Prevention of Pollution by Harmful Substances Carried by Sea in Packaged Form
Annex IV - Prevention of Pollution by Sewage from Ships
Annex V - Prevention of Pollution by Garbage from Ships
Annex VI - Prevention of Air Pollution from Ships

INTERNATIONAL CONVENTION ON STANDARDS OF TRAINING, CERTIFICATION AND WATCHKEEPING FOR SEAFARERS (STCW), 1978

STCW sets qualification standards for masters, officers and watch personnel on seagoing merchant ships.

Key chapters:
Chapter I - General provisions
Chapter II - Master and deck department
Chapter III - Engine department
Chapter IV - Radiocommunication and radio personnel
Chapter V - Special training requirements for personnel on certain types of ships
Chapter VI - Emergency, occupational safety, medical care and survival functions
Chapter VII - Alternative certification
Chapter VIII - Watchkeeping
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved IMO conventions summary ({len(text)} chars)")
    return text


def collect_international_shipping_regulations():
    out = RAW_DIR / "international_shipping_regulations.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """中华人民共和国国际海运条例

（2001年12月11日国务院令第335号公布，根据2013年7月18日国务院令第638号、2016年2月6日国务院令第666号、2019年3月2日国务院令第709号修订）

第一章　总则

第一条　为了规范国际海上运输活动，维护国际海运市场秩序，保障国际海运各方当事人的合法权益，促进国际海运业的健康发展，制定本条例。

第二条　本条例适用于进出中华人民共和国港口的国际海上运输经营活动以及与国际海上运输相关的辅助性经营活动。

第三条　从事国际海上运输经营活动以及与国际海上运输相关的辅助性经营活动，应当遵循诚实信用的原则，依法经营，公平竞争。

第四条　国务院交通运输主管部门主管全国的国际海运事业。

第二章　国际海上运输及其辅助性业务的经营者

第五条　经营国际船舶运输业务，应当具备下列条件：
（一）有与经营国际海上运输业务相适应的船舶，其中必须有中国籍船舶；
（二）投入运营的船舶符合国家规定的海上交通安全技术标准；
（三）有提单、客票或者多式联运单证；
（四）有具备国务院交通运输主管部门规定的从业资格的高级业务管理人员。

第六条　经营国际船舶运输业务，应当向国务院交通运输主管部门提出申请，并附送符合本条例第五条规定条件的相关材料。

第七条　经营无船承运业务，应当向国务院交通运输主管部门办理提单登记。

第八条　经营国际船舶代理业务，应当具备下列条件：
（一）高级业务管理人员中至少2人具有3年以上从事国际海上运输经营活动的经历；
（二）有固定的营业场所和必要的营业设施。

第九条　经营国际船舶管理业务，应当具备下列条件：
（一）高级业务管理人员中至少2人具有3年以上从事国际海上运输经营活动的经历；
（二）有固定的营业场所和必要的营业设施。

第十条　中国的船舶运输经营者在中国境内设立外商投资企业，适用国家有关外商投资的法律、行政法规。

第三章　国际海上运输的经营活动

第十一条　国际船舶运输经营者经营进出中国港口的国际班轮运输业务，应当依照本条例的规定取得国际班轮运输经营资格。

第十二条　经营国际班轮运输业务，应当向国务院交通运输主管部门提出申请。

第十三条　国际班轮运输经营者应当将经营的国际班轮航线信息向国务院交通运输主管部门备案。

第十四条　国际船舶运输经营者不得以低于正常、合理水平的运价提供服务，妨碍公平竞争。

第十五条　国际船舶运输经营者不得以给定的运费或者其他条件获取不公平的竞争优势。

第十六条　国际船舶运输经营者不得滥用优势地位，以歧视性价格或者其他限制性条件限制其他经营者。

第四章　海运协定和国际合作

第十七条　中华人民共和国政府依照缔结或者参加的国际条约和国际惯例，给予外国国际船舶运输经营者与国内经营者同等的待遇。

第十八条　国务院交通运输主管部门应当与外国政府和相关国际组织进行协调，促进国际海运合作。

第五章　调查和处理

第十九条　国务院交通运输主管部门可以对国际海运市场秩序和公平竞争情况进行调查。

第二十条　调查处理应当遵循公平、公正的原则。

第二十一条　被调查人应当配合调查，如实提供有关情况和资料。

第六章　法律责任

第二十二条　违反本条例的规定，未经许可擅自经营国际船舶运输业务的，由国务院交通运输主管部门责令停止经营。

第二十三条　违反本条例的规定，国际船舶运输经营者未按规定备案的，由国务院交通运输主管部门责令限期改正。

第二十四条　违反本条例的规定，国际船舶运输经营者以低于正常、合理水平的运价提供服务的，由国务院交通运输主管部门责令改正。

第二十五条　违反本条例的规定，国际船舶运输经营者滥用优势地位的，由国务院交通运输主管部门责令改正。

第七章　附则

第二十六条　本条例自2002年1月1日起施行。
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved international shipping regulations ({len(text)} chars)")
    return text


def collect_port_law():
    out = RAW_DIR / "port_law.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """中华人民共和国港口法

（2003年6月28日第十届全国人民代表大会常务委员会第三次会议通过，根据2018年12月29日第十三届全国人民代表大会常务委员会第七次会议《关于修改〈中华人民共和国电力法〉等四部法律的决定》修正）

第一章　总则

第一条　为了加强港口管理，维护港口的安全与经营秩序，保护当事人的合法权益，促进港口的建设与发展，制定本法。

第二条　从事港口规划、建设、维护、经营、管理及其相关活动，适用本法。

第三条　本法所称港口，是指具有船舶进出、停泊、靠泊，旅客上下，货物装卸、驳运、储存等功能，具有相应的码头设施，由一定范围的水域和陆域组成的区域。
港口可以由一个或者多个港区组成。

第四条　国务院和有关县级以上地方人民政府应当在国民经济和社会发展计划中体现港口的发展和规划要求，并依法保护港口合理利用。

第五条　国家鼓励国内外经济组织和个人依法投资建设、经营港口，保护投资者的合法权益。

第六条　国务院交通运输主管部门主管全国的港口工作。
地方人民政府对本行政区域内港口的管理，按照国务院关于港口管理体制的规定确定。
依照前款确定的港口管理体制，由港口所在地的港口行政管理部门对港口实行统一管理。

第二章　港口规划与建设

第七条　港口规划应当根据国民经济和社会发展的要求以及国防建设的需要编制，体现合理利用岸线资源的原则，符合城镇体系规划，并与土地利用总体规划、城市总体规划、江河流域规划、防洪规划、海洋功能区划、水路运输发展规划和其他运输方式发展规划以及法律、行政法规的规定相衔接。

第八条　港口规划包括港口布局规划和港口总体规划。
港口布局规划，是指港口的分布规划，包括全国港口布局规划和省、自治区、直辖市港口布局规划。
港口总体规划，是指一个港口在一定时期内的具体规划。

第九条　全国港口布局规划，由国务院交通运输主管部门征求国务院有关部门和有关军事机关的意见编制，报国务院批准后公布实施。
省、自治区、直辖市港口布局规划，由省、自治区、直辖市人民政府根据全国港口布局规划组织编制，并送国务院交通运输主管部门征求意见。

第十条　港口总体规划由港口行政管理部门征求有关部门和有关军事机关的意见编制。

第十一条　港口总体规划应当符合港口布局规划。

第十二条　港口建设应当符合港口规划。不得违反港口规划建设任何港口设施。

第十三条　按照国家规定须经有关机关批准的港口建设项目，应当按照国家有关规定办理审批手续。

第十四条　港口建设应当使用符合国家和行业标准的港口设施。

第三章　港口经营

第十五条　从事港口经营，应当向港口行政管理部门书面申请取得港口经营许可，并依法办理工商登记。

第十六条　港口经营人应当具备下列条件：
（一）有固定的经营场所；
（二）有与经营业务相适应的设施、设备；
（三）有与经营业务相适应的专业技术人员和管理人员；
（四）法律、法规规定的其他条件。

第十七条　港口经营人从事经营活动，必须遵守有关法律、法规，遵守国务院交通运输主管部门有关港口作业规则的规定，依法履行合同约定的义务，为客户提供公平、良好的服务。

第十八条　港口经营人应当依照有关环境保护的法律、法规的规定，采取有效措施，防治对环境的污染和危害。

第十九条　港口经营人应当优先安排抢险物资、救灾物资和国防建设急需物资的作业。

第二十条　港口经营人应当依照有关安全生产的法律、法规的规定，加强安全生产管理，建立健全安全生产责任制，完善安全生产条件，确保安全生产。

第二十一条　港口经营人应当依法制定本单位的危险货物事故应急预案、重大生产安全事故的旅客紧急疏散和救援预案。

第二十二条　港口行政管理部门应当依法制定可能危及社会公共利益的港口危险货物事故应急预案、重大生产安全事故的旅客紧急疏散和救援预案。

第四章　港口安全

第二十三条　港口行政管理部门应当依法对港口安全生产情况实施监督检查，对旅客上下集中、货物装卸量较大或者有特殊用途的码头进行重点巡查。

第二十四条　港口经营人应当依照有关法律、法规和国务院交通运输主管部门有关港口安全作业规则的规定，加强港口安全生产。

第二十五条　在港口内进行危险货物的装卸、过驳作业，港口经营人应当按照有关规定将危险货物的名称、特性、包装和作业的时间、地点报告港口行政管理部门。

第二十六条　船舶进出港口，应当依照有关水上交通安全的法律、行政法规的规定向海事管理机构报告。

第二十七条　港口行政管理部门应当依法制定港口危险货物事故应急预案、重大生产安全事故的旅客紧急疏散和救援预案。

第二十八条　港口行政管理部门应当对港口经营人制定的应急预案以及实施情况进行监督检查。

第五章　法律责任

第二十九条　违反本法规定，未经依法批准，建设港口设施使用港口岸线的，由县级以上地方人民政府或者港口行政管理部门责令限期改正。

第三十条　违反本法规定，港口经营人不优先安排抢险物资、救灾物资、国防建设急需物资的作业的，由港口行政管理部门责令改正。

第三十一条　违反本法规定，港口经营人未依法制定应急预案或者未按照应急预案采取应急处置措施的，由港口行政管理部门责令改正。

第三十二条　违反本法规定，港口经营人未依法报告危险货物作业情况的，由港口行政管理部门责令改正。

第三十三条　港口行政管理部门不依法履行职责，对直接负责的主管人员和其他直接责任人员依法给予行政处分。

第六章　附则

第三十四条　本法自2004年1月1日起施行。
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved port law ({len(text)} chars)")
    return text


def collect_inland_waterway_safety():
    out = RAW_DIR / "inland_waterway_safety.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """中华人民共和国内河交通安全管理条例

（2002年6月28日国务院令第355号公布，根据2011年1月8日国务院令第588号、2017年3月1日国务院令第676号、2019年3月2日国务院令第709号修订）

第一章　总则

第一条　为了加强内河交通安全管理，维护内河交通秩序，保障人民群众生命、财产安全，制定本条例。

第二条　在中华人民共和国内河通航水域从事航行、停泊和作业以及与内河交通安全有关的活动，必须遵守本条例。

第三条　内河交通安全管理遵循安全第一、预防为主、方便群众、依法管理的原则。

第四条　国务院交通运输主管部门主管全国内河交通安全管理工作。

第二章　船舶、浮动设施和船员

第五条　船舶应当符合国家有关安全技术规范，经船舶检验机构检验合格，取得船舶检验证书，方可航行。

第六条　浮动设施应当经船舶检验机构检验合格，取得浮动设施检验证书。

第七条　船舶应当按照国家规定配备船员。

第八条　船员应当经过专业培训，持有有效的船员适任证书。

第三章　航行、停泊和作业

第九条　船舶在内河航行，应当悬挂国旗，标明船名、船籍港、载重线。

第十条　船舶在内河航行，应当遵守航行规则，保持安全航速。

第十一条　船舶在内河停泊，应当选择安全的停泊地点。

第十二条　从事水上水下作业，应当事先报经海事管理机构批准。

第四章　危险货物运输

第十三条　船舶载运危险货物，必须符合国家有关危险货物运输的安全管理规定。

第十四条　船舶载运危险货物进出港口，应当在进出港口24小时前向海事管理机构报告。

第十五条　禁止装运法律、行政法规禁止运输的危险货物。

第五章　渡口管理

第十六条　渡口的设置应当符合国家有关规定，并报经县级以上地方人民政府批准。

第十七条　渡口经营者应当对渡口安全负责，建立健全安全管理制度。

第十八条　渡船应当按照国家规定配备救生、消防设备。

第六章　通航保障

第十九条　航道应当保持良好的通航状态。

第二十条　任何单位和个人不得破坏、损坏或者非法占用航道。

第二十一条　桥梁、隧道等跨河建筑物应当设置必要的助航标志。

第七章　救助和事故处理

第二十二条　船舶在内河遇险，应当采取一切有效措施自救。

第二十三条　海事管理机构接到遇险报告后，应当立即组织救助。

第二十四条　内河交通事故的报告、调查和处理，依照国家有关规定执行。

第八章　法律责任

第二十五条　违反本条例的规定，未经检验擅自航行的，由海事管理机构责令停止航行。

第二十六条　违反本条例的规定，船舶超载运输的，由海事管理机构责令改正。

第二十七条　违反本条例的规定，船舶违法载运危险货物的，由海事管理机构责令停止违法行为。

第九章　附则

第二十八条　本条例自2002年8月1日起施行。
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved inland waterway safety ({len(text)} chars)")
    return text


def collect_ship_pollution_prevention():
    out = RAW_DIR / "ship_pollution_prevention.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """中华人民共和国防治船舶污染海洋环境管理条例

（2009年9月9日国务院令第561号公布，根据2013年7月18日国务院令第638号、2014年7月29日国务院令第653号、2016年2月6日国务院令第666号、2017年3月1日国务院令第676号、2018年3月19日国务院令第698号修订）

第一章　总则

第一条　为了防治船舶及其有关作业活动污染海洋环境，保护海洋生态环境，促进经济社会可持续发展，制定本条例。

第二条　本条例适用于在中华人民共和国管辖海域内防治船舶及其有关作业活动污染海洋环境的管理。

第三条　防治船舶及其有关作业活动污染海洋环境，实行预防为主、防治结合、污染担责的原则。

第四条　国务院交通运输主管部门主管全国防治船舶及其有关作业活动污染海洋环境工作。

第二章　一般规定

第五条　船舶的结构、设备、器材应当符合国家有关防治船舶污染海洋环境的规范和要求。

第六条　船舶应当按照国家规定持有防治船舶污染海洋环境的证书。

第七条　船舶应当按照国家规定对船舶油污水、生活污水、垃圾等进行收集、处理。

第八条　船舶应当建立并运行船舶污染防治管理制度。

第三章　船舶污染物的排放和接收

第九条　船舶不得违反规定向海洋排放污染物。

第十条　船舶需要接收污染物的，应当委托依法设立的船舶污染物接收单位接收。

第十一条　船舶污染物接收单位应当具备相应的接收能力和条件。

第十二条　船舶污染物接收作业应当遵守有关安全和防污染操作规程。

第四章　船舶有关作业活动的污染防治

第十三条　船舶进行洗舱、清舱、驱气作业，应当采取有效的安全和防污染措施。

第十四条　船舶进行加油作业，应当遵守有关安全操作规程。

第十五条　船舶进行油漆、除锈等作业，应当采取有效的防污染措施。

第五章　船舶污染事故应急处置

第十六条　国务院交通运输主管部门应当组织制定国家船舶污染事故应急预案。

第十七条　船舶发生污染事故，应当立即启动应急预案，采取有效措施控制和消除污染。

第十八条　船舶污染事故的报告、调查和处理，依照国家有关规定执行。

第六章　法律责任

第十九条　违反本条例的规定，船舶向海洋排放污染物的，由海事管理机构处以罚款。

第二十条　违反本条例的规定，船舶未持有有效防污证书的，由海事管理机构责令改正。

第二十一条　违反本条例的规定，船舶污染物接收单位未具备接收能力的，由海事管理机构责令改正。

第七章　附则

第二十二条　本条例自2010年3月1日起施行。
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved ship pollution prevention ({len(text)} chars)")
    return text


def collect_detailed_solas():
    out = RAW_DIR / "solas_detailed.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """INTERNATIONAL CONVENTION FOR THE SAFETY OF LIFE AT SEA (SOLAS), 1974 - DETAILED EXTRACTS

CHAPTER I - GENERAL PROVISIONS
Regulation 1: Application
(a) The present regulations shall apply to ships entitled to fly the flag of any Contracting Government.
(b) The classes of ships to which each chapter applies are specified in each chapter.

Regulation 6: Inspection and Survey
The inspection and survey of ships shall be carried out by officers of the Administration. The Administration may entrust the inspections and surveys either to surveyors nominated for the purpose or to recognized organizations.

Regulation 7: Survey of Passenger Ships
Passenger ships shall be subject to surveys: an initial survey before the ship is put in service; a periodical survey once every 12 months; and additional surveys as necessary.

Regulation 8: Survey of Cargo Ships
The cargo ship safety equipment survey shall be carried out at intervals not exceeding 5 years. The cargo ship safety construction survey shall be carried out at intervals not exceeding 5 years.

Regulation 10: Surveys of Hull, Machinery and Equipment
The surveys shall be such as to ensure that the hull, machinery and equipment are in satisfactory condition and fit for the service for which the ship is intended.

Regulation 11: Maintenance of Conditions after Survey
After any survey the condition of the ship and its equipment shall be maintained to conform with the provisions of the regulations.

Regulation 16: Maintenance of Certificates
No ship shall proceed to sea without valid certificates.

Regulation 19: Control
Every ship while in the port of another Contracting Government is subject to control by officers duly authorized by such Government.

CHAPTER II-1 - CONSTRUCTION - SUBDIVISION AND STABILITY, MACHINERY AND ELECTRICAL INSTALLATIONS
Regulation 1: Application
This chapter applies to ships the keels of which are laid on or after 1 July 1998, unless expressly provided otherwise.

Regulation 2: Definitions
"Subdivision length" means the length of the ship measured at the deepest subdivision load line.

Regulation 3: Intact Stability
The intact stability of passenger ships shall be determined by a stability investigation.

Regulation 5: Permeability
The permeability of a space shall be determined as follows: for spaces occupied by passengers, 95%; for spaces occupied by cargo, 60%; for spaces occupied by stores, 60%; for machinery spaces, 85%.

Regulation 7: Peak and Machinery Space Bulkheads
Collision bulkheads shall be fitted which shall be watertight up to the freeboard deck.

Regulation 8: Double Bottoms
Double bottoms shall be fitted in passenger ships and in cargo ships of 5,000 tons gross tonnage and upwards.

Regulation 9: Assignment of Subdivision Load Lines
The subdivision load lines assigned and marked on the side of a passenger ship shall be recorded in the Passenger Ship Safety Certificate.

CHAPTER II-2 - FIRE PROTECTION
Regulation 1: Application
This chapter applies to ships constructed on or after 1 July 2002.

Regulation 2: Definitions
"Non-combustible material" means a material which neither burns nor gives off flammable vapours in sufficient quantity for self-ignition when heated to 750°C.

Regulation 3: Fire Safety Objectives
The fire safety objectives are to prevent fire; to detect and extinguish fire; to prevent the spread of fire; to provide means of escape.

Regulation 4: Probability of Ignition
The probability of ignition of a space shall be minimized by limiting the quantity of combustible materials.

Regulation 5: Fire Growth Potential
The potential for fire growth shall be limited by the use of non-combustible materials and the installation of fire detection and alarm systems.

Regulation 6: Smoke Generation Potential
Smoke generation potential from combustible materials shall be limited.

Regulation 7: Detection and Alarm
An automatic fire detection and alarm system shall be installed in machinery spaces and accommodation spaces.

Regulation 8: Control of Smoke Spread
Smoke from a fire shall be controlled by the installation of smoke extraction systems.

Regulation 9: Containment of Fire
Fire shall be contained by the use of fire-resisting divisions.

Regulation 10: Means of Escape
Means of escape shall be provided for all spaces to allow passengers and crew to readily evacuate the ship.

CHAPTER III - LIFE-SAVING APPLIANCES
Regulation 1: Application
This chapter applies to all ships.

Regulation 2: Definitions
"Lifeboat" means a boat capable of sustaining the lives of persons in distress from the time of abandoning the ship.

Regulation 3: Evaluation and Testing
Life-saving appliances shall be evaluated by testing as required by the Administration.

Regulation 4: Lifeboats
Sufficient lifeboats shall be provided on each side of the ship to accommodate all persons on board.

Regulation 5: Rescue Boats
Ships shall carry at least one rescue boat for retrieving persons in distress.

Regulation 6: Life-rafts
Life-rafts shall be provided in sufficient number to accommodate all persons on board.

Regulation 7: Lifebuoys
Lifebuoys shall be provided in accordance with the requirements of the Life-Saving Appliance Code.

Regulation 8: Lifejackets
A lifejacket shall be provided for every person on board.

Regulation 9: Immersion Suits
Immersion suits shall be provided for all persons on board on ships operating in cold climates.

Regulation 10: Muster List and Emergency Instructions
Muster lists and emergency instructions shall be posted in conspicuous places throughout the ship.

CHAPTER IV - RADIOCOMMUNICATIONS
Regulation 1: Application
This chapter applies to ships to which chapter I applies.

Regulation 2: Terms and Definitions
"Global Maritime Distress and Safety System (GMDSS)" means the system established by IMO.

Regulation 3: Radio Installations
Ships shall be provided with radio installations capable of performing distress and safety communications.

Regulation 4: Maintenance
Radio equipment shall be maintained to ensure availability of the functional requirements.

CHAPTER V - SAFETY OF NAVIGATION
Regulation 1: Application
This chapter applies to all ships on all voyages.

Regulation 2: Definitions
"Master" means the person having command of a ship.

Regulation 3: Navigational Equipment
Ships shall be fitted with navigational equipment including radar, compass, echo sounder, and speed and distance measuring device.

Regulation 4: Automatic Identification System (AIS)
Ships shall be fitted with an Automatic Identification System (AIS).

Regulation 5: Voyage Data Recorder (VDR)
Ships shall be fitted with a Voyage Data Recorder (VDR).

Regulation 6: Ship Reporting Systems
Ship reporting systems shall be used to contribute to safety of life at sea.

Regulation 7: Dangerous Messages
The master of every ship shall communicate to nearby ships and authorities any danger to navigation.

Regulation 8: Distress Messages
The master of a ship in distress shall send distress signals.

Regulation 9: Communication
Ships shall maintain continuous listening watch on VHF channel 16.

Regulation 10: Search and Rescue
The master of every ship receiving a distress signal shall proceed with all speed to assist.

CHAPTER VI - CARRIAGE OF CARGOES
Regulation 1: Application
This chapter applies to the carriage of cargoes which because of their particular hazards may pose a risk to ships or persons.

Regulation 2: Cargo Information
The shipper shall provide the master with appropriate information on the cargo.

Regulation 3: Stowage and Securing
Cargo shall be stowed and secured to prevent throughout the voyage damage or hazard to the ship and persons.

Regulation 4: Bulk Cargoes
Bulk cargoes shall be loaded and carried in accordance with the requirements of the IMSBC Code.

CHAPTER VII - CARRIAGE OF DANGEROUS GOODS
Regulation 1: Application
The carriage of dangerous goods shall be in compliance with the IMDG Code.

Regulation 2: Documentation
Every ship carrying dangerous goods shall have a special list or manifest setting forth the dangerous goods.

Regulation 3: Stowage
Dangerous goods shall be properly stowed and secured.

CHAPTER IX - MANAGEMENT FOR THE SAFE OPERATION OF SHIPS (ISM CODE)
Regulation 1: Definitions
"International Safety Management (ISM) Code" means the International Management Code for the Safe Operation of Ships.

Regulation 2: Application
The ISM Code applies to all ships of 500 gross tonnage and above.

Regulation 3: Safety Management Requirements
The Company shall establish a Safety Management System (SMS) including safety and environmental protection policy.

Regulation 4: Certification
The ship shall be operated by a Company holding a Document of Compliance (DOC) and the ship shall hold a Safety Management Certificate (SMC).

CHAPTER XI-2 - SPECIAL MEASURES TO ENHANCE MARITIME SECURITY (ISPS CODE)
Regulation 1: Definitions
"International Ship and Port Facility Security (ISPS) Code" means the International Code for the Security of Ships and of Port Facilities.

Regulation 2: Application
This chapter applies to ships engaged on international voyages of 500 gross tonnage and above.

Regulation 3: Company and Ship Security
The Company shall designate a Company Security Officer (CSO) and the ship shall have a Ship Security Officer (SSO).

Regulation 4: Ship Security Plan
A Ship Security Plan (SSP) shall be approved for each ship.

Regulation 5: Verification and Certification
Ships shall be verified and certified in accordance with the ISPS Code.

CHAPTER XII - ADDITIONAL SAFETY MEASURES FOR BULK CARRIERS
Regulation 1: Definitions
"Bulk carrier" means a ship constructed with a single deck, top-side tanks and hopper side tanks.

Regulation 6: Damage Assumptions
Bulk carriers shall be capable of surviving flooding of any one cargo hold.

CHAPTER XIV - SAFETY MEASURES FOR SHIPS OPERATING IN POLAR WATERS
Regulation 1: Definitions
"Polar Code" means the International Code for Ships Operating in Polar Waters.

Regulation 2: Application
This chapter applies to ships operating in polar waters.
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved detailed SOLAS extracts ({len(text)} chars)")
    return text


def collect_detailed_marpol():
    out = RAW_DIR / "marpol_detailed.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """INTERNATIONAL CONVENTION FOR THE PREVENTION OF POLLUTION FROM SHIPS (MARPOL 73/78) - DETAILED EXTRACTS

GENERAL OBLIGATIONS
The Parties to the Convention undertake to give effect to the provisions of the present Convention and those Annexes thereto by which they are bound, in order to prevent the pollution of the marine environment by the discharge of harmful substances or effluents containing such substances in contravention of the Convention.

Definitions:
"Discharge" means any release from a ship, including any escape, disposal, spilling, leaking, pumping, emitting or emptying.
"Oil" means petroleum in any form including crude oil, fuel oil, sludge, oil refuse and refined products.
"Harmful substance" means any substance which, if introduced into the sea, is liable to create hazards to human health, to harm living resources and marine life.

ANNEX I - REGULATIONS FOR THE PREVENTION OF POLLUTION BY OIL
Regulation 1: Definitions
"Oil tanker" means a ship constructed or adapted primarily to carry oil in bulk.
"Oil residue (sludge)" means the residual waste oil products generated during the normal operation of a ship.

Regulation 12: Tanks for Oil Residues (Sludge)
Every ship of 400 gross tonnage and above shall be provided with a tank or tanks of adequate capacity for oil residues.

Regulation 13: Oil Filtering Equipment
Ships of 400 gross tonnage and above shall be fitted with oil filtering equipment.

Regulation 14: Segregated Ballast Tanks
Crude oil tankers of 20,000 tons deadweight and above and product carriers of 30,000 tons deadweight and above shall be provided with segregated ballast tanks.

Regulation 15: Oil Discharge
Any discharge into the sea of oil or oily mixtures shall be prohibited except when certain conditions are met: the ship is proceeding en route; the oil content without dilution does not exceed 15 ppm; and the ship has in operation oil filtering equipment.

Regulation 16: Oil Record Book
Every oil tanker and every ship of 400 gross tonnage and above shall maintain an Oil Record Book.

Regulation 20: Oil Discharge Monitoring
Oil tankers shall be fitted with an oil discharge monitoring and control system.

Regulation 21: Subdivision and Stability
Oil tankers shall comply with subdivision and stability requirements.

ANNEX II - REGULATIONS FOR THE CONTROL OF POLLUTION BY NOXIOUS LIQUID SUBSTANCES IN BULK
Regulation 1: Definitions
"Noxious liquid substance" means any substance designated in the IBC Code.

Regulation 3: Categorization
Noxious liquid substances are categorized as X, Y, or Z based on their hazard to marine resources.

Category X: substances which present a major hazard to marine resources - discharge prohibited.
Category Y: substances which present a hazard to marine resources - discharge limited.
Category Z: substances which present a minor hazard to marine resources - discharge subject to minimal controls.

Regulation 5: Discharge of Residues
Discharge of residues of noxious liquid substances shall comply with specific concentration limits and ship speed requirements.

Regulation 8: Procedures and Arrangements
Ships carrying noxious liquid substances shall have a Procedures and Arrangements Manual.

ANNEX III - PREVENTION OF POLLUTION BY HARMFUL SUBSTANCES CARRIED BY SEA IN PACKAGED FORM
Regulation 1: Application
This Annex applies to all ships carrying harmful substances in packaged form.

Regulation 2: Packing
Packages shall be adequate to minimize the hazard to the marine environment.

Regulation 3: Marking and Labeling
Packages shall be marked with the correct technical name and labeled.

Regulation 4: Documentation
Ships carrying harmful substances shall have a special list or manifest.

Regulation 5: Stowage
Harmful substances shall be properly stowed and secured.

ANNEX IV - PREVENTION OF POLLUTION BY SEWAGE FROM SHIPS
Regulation 1: Definitions
"Sewage" means drainage from toilets, urinals, and medical premises.

Regulation 2: Application
This Annex applies to ships of 400 gross tonnage and above.

Regulation 3: Sewage Treatment Plant
Ships shall be equipped with a sewage treatment plant or sewage comminuting and disinfecting system.

Regulation 4: Discharge of Sewage
Discharge of sewage into the sea is prohibited unless the ship has in operation an approved sewage treatment plant.

ANNEX V - PREVENTION OF POLLUTION BY GARBAGE FROM SHIPS
Regulation 1: Definitions
"Garbage" means all kinds of food wastes, domestic wastes and operational wastes.

Regulation 2: Application
This Annex applies to all ships.

Regulation 3: Disposal of Garbage
Disposal of all plastics into the sea is prohibited. Disposal of food wastes is permitted beyond 12 nautical miles from the nearest land.

Regulation 4: Garbage Management Plan
Every ship of 100 gross tonnage and above shall carry a Garbage Management Plan.

Regulation 5: Garbage Record Book
Ships of 400 gross tonnage and above shall maintain a Garbage Record Book.

ANNEX VI - PREVENTION OF AIR POLLUTION FROM SHIPS
Regulation 1: Application
This Annex applies to all ships.

Regulation 2: Definitions
"Emission" means the release of substances from the ship into the atmosphere.

Regulation 12: Ozone-Depleting Substances
Emissions of ozone-depleting substances are prohibited.

Regulation 13: Nitrogen Oxides (NOx)
Marine diesel engines installed on ships shall comply with NOx emission standards.

Regulation 14: Sulphur Oxides (SOx)
The sulphur content of fuel oil used on board ships shall not exceed specified limits. From 1 January 2020, the sulphur content shall not exceed 0.50% m/m outside Emission Control Areas.

Regulation 15: Volatile Organic Compounds
Tankers carrying crude oil shall have a vapour emission control system.

Regulation 16: Shipboard Incineration
Shipboard incineration is prohibited except for specified materials.

Regulation 18: Fuel Oil Quality
Fuel oil shall not exceed the specified sulphur content and shall be documented by a Bunker Delivery Note.

Regulation 19: Energy Efficiency
Ships shall comply with the Energy Efficiency Design Index (EEDI) requirements.

Regulation 20: Carbon Intensity
Ships shall comply with the Carbon Intensity Indicator (CII) requirements.

Regulation 22: Fuel Oil Consumption Data
Ships shall collect and report fuel oil consumption data.
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved detailed MARPOL extracts ({len(text)} chars)")
    return text


def collect_detailed_stcw():
    out = RAW_DIR / "stcw_detailed.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """INTERNATIONAL CONVENTION ON STANDARDS OF TRAINING, CERTIFICATION AND WATCHKEEPING FOR SEAFARERS (STCW), 1978 - DETAILED EXTRACTS

CHAPTER I - GENERAL PROVISIONS
Regulation I/1: Definitions
"Certificate" means a valid document issued by the Administration authorizing the holder to serve as stated.
"Competent authority" means the authority designated by the Party to issue certificates.

Regulation I/2: Certificates and Endorsements
Certificates shall be issued in accordance with the Convention to masters, officers and ratings.

Regulation I/3: Principles Governing Near-coastal Voyages
Parties may establish provisions for near-coastal voyages.

Regulation I/4: Control Procedures
A ship in a port of another Party may be inspected to verify that all personnel hold appropriate certificates.

Regulation I/5: National Provisions
Parties shall establish processes for the investigation of incompetence, acts, omissions or fraud.

Regulation I/6: Training and Assessment
Parties shall ensure that all training and assessment is conducted by qualified personnel.

Regulation I/7: Communication of Information
Parties shall communicate to the IMO information concerning training and certification.

Regulation I/8: Quality Standards
Parties shall ensure that training, assessment and certification systems are evaluated by quality standards.

Regulation I/9: Medical Standards
Parties shall establish medical standards for seafarers.

Regulation I/10: Recognition of Certificates
Parties may recognize certificates issued by other Parties.

Regulation I/11: Validity of Certificates
Certificates shall be issued for a period specified by the Administration.

CHAPTER II - MASTER AND DECK DEPARTMENT
Regulation II/1: Minimum Requirements for Certification of Officers in Charge of a Navigational Watch
Every officer in charge of a navigational watch shall hold a certificate of competency.

Regulation II/2: Minimum Requirements for Certification of Masters and Chief Mates
Every master and chief mate on ships of 500 gross tonnage and above shall hold a certificate of competency.

Regulation II/3: Minimum Requirements for Certification of Officers in Charge of a Navigational Watch on Ships of Less Than 500 Gross Tonnage
Officers on smaller ships shall hold appropriate certificates.

Regulation II/4: Minimum Requirements for Certification of Ratings Forming Part of a Navigational Watch
Ratings forming part of a navigational watch shall be certified.

CHAPTER III - ENGINE DEPARTMENT
Regulation III/1: Minimum Requirements for Certification of Officers in Charge of an Engineering Watch
Every officer in charge of an engineering watch shall hold a certificate of competency.

Regulation III/2: Minimum Requirements for Certification of Chief Engineer Officers and Second Engineer Officers
Chief engineer officers and second engineer officers on ships of 3,000 kW propulsion power and above shall hold certificates of competency.

Regulation III/3: Minimum Requirements for Certification of Chief Engineer Officers and Second Engineer Officers on Ships of 750 kW to 3,000 kW
Engine officers on smaller ships shall hold appropriate certificates.

Regulation III/4: Minimum Requirements for Certification of Ratings Forming Part of an Engineering Watch
Ratings forming part of an engineering watch shall be certified.

CHAPTER IV - RADIOCOMMUNICATION AND RADIO PERSONNEL
Regulation IV/1: Application
This chapter applies to personnel on ships operating in the GMDSS.

Regulation IV/2: Minimum Requirements for Certification of GMDSS Radio Personnel
Every person in charge of radio communications shall hold a GMDSS certificate.

CHAPTER V - SPECIAL TRAINING REQUIREMENTS
Regulation V/1: Training for Personnel on Certain Types of Ships
Special training is required for personnel on tankers, ro-ro passenger ships, and other specialized vessels.

Regulation V/2: Training for Personnel on Passenger Ships
Personnel on passenger ships shall receive training in crowd management and crisis management.

Regulation V/3: Training for Personnel on Tankers
Personnel on oil tankers, chemical tankers and liquefied gas tankers shall hold a certificate of specialized training.

CHAPTER VI - EMERGENCY, OCCUPATIONAL SAFETY AND MEDICAL CARE
Regulation VI/1: Minimum Training in Personal Survival Techniques
All seafarers shall receive basic training in personal survival techniques.

Regulation VI/2: Minimum Training in Fire Prevention and Fire Fighting
All seafarers shall receive basic training in fire prevention and fire fighting.

Regulation VI/3: Minimum Training in Elementary First Aid
All seafarers shall receive basic training in elementary first aid.

Regulation VI/4: Minimum Training in Personal Safety and Social Responsibilities
All seafarers shall receive basic training in personal safety and social responsibilities.

Regulation VI/5: Medical Care
Seafarers designated to provide medical care shall hold a certificate in medical care.

CHAPTER VII - ALTERNATIVE CERTIFICATION
Regulation VII/1: Alternative Certification
Parties may issue alternative certificates that provide equivalent competencies.

Regulation VII/2: Principles of Alternative Certification
Alternative certification shall ensure a level of competence equivalent to that required by the Convention.

CHAPTER VIII - WATCHKEEPING
Regulation VIII/1: Fitness for Duty
Administrations shall require that watchkeeping personnel are rested and fit for duty.

Regulation VIII/2: Watchkeeping Arrangements
Watchkeeping schedules shall be posted and observed.

Regulation VIII/3: Prevention of Alcohol Abuse
Administrations shall establish limits for alcohol consumption.
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved detailed STCW extracts ({len(text)} chars)")
    return text


def collect_ism_code():
    out = RAW_DIR / "ism_code.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """INTERNATIONAL MANAGEMENT CODE FOR THE SAFE OPERATION OF SHIPS AND FOR POLLUTION PREVENTION (ISM CODE)

PREAMBLE
The purpose of this Code is to provide an international standard for the safe management and operation of ships and for pollution prevention.

OBJECTIVES
The objectives of the Code are to ensure safety at sea, prevention of human injury or loss of life, and avoidance of damage to the environment, particularly the marine environment.

APPLICATION
The Code applies to all ships of 500 gross tonnage and above.

FUNCTIONAL REQUIREMENTS
Every Company shall develop, implement and maintain a Safety Management System (SMS) which includes the following functional requirements:
1. A safety and environmental protection policy
2. Instructions and procedures to ensure safe operation of ships
3. Defined levels of authority and lines of communication
4. Procedures for reporting accidents and non-conformities
5. Procedures for emergency preparedness and response
6. Procedures for internal audits and management reviews

SECTION A - IMPLEMENTATION
1. General
   1.1 The Company should document and maintain its safety management objectives.
   1.2 The Company should establish a safety management system.

2. Safety and Environmental Protection Policy
   2.1 The Company shall establish a safety and environmental protection policy.
   2.2 The policy shall be implemented and maintained at all levels.

3. Company Responsibilities and Authority
   3.1 The Company shall define the responsibility, authority and interrelation of personnel who manage, perform and verify work.
   3.2 The Company shall provide necessary resources and shore-based support.

4. Designated Person(s)
   4.1 The Company shall designate a person or persons ashore having direct access to the highest level of management.
   4.2 The designated person shall monitor the safety and pollution prevention aspects of each ship's operation.

5. Master's Responsibility and Authority
   5.1 The Company shall define the master's responsibility regarding the safety management system.
   5.2 The master has the overriding authority and responsibility to make decisions regarding safety and pollution prevention.

6. Resources and Personnel
   6.1 The Company shall ensure that the master is properly qualified and fully conversant with the SMS.
   6.2 The Company shall ensure that each ship is manned with qualified, certificated and medically fit seafarers.
   6.3 The Company shall establish procedures to ensure that new personnel are familiarized with their duties.

7. Development of Plans for Shipboard Operations
   7.1 The Company shall establish procedures for the preparation of plans and instructions for key shipboard operations.

8. Emergency Preparedness
   8.1 The Company shall establish procedures to identify, describe and respond to potential emergency situations.
   8.2 The Company shall establish programs for drills and exercises.

9. Reports and Analysis of Non-Conformities, Accidents and Hazardous Occurrences
   9.1 The SMS shall include procedures for reporting non-conformities, accidents and hazardous occurrences.
   9.2 The Company shall investigate and analyze such events to improve safety.

10. Maintenance of the Ship and Equipment
    10.1 The Company shall establish procedures to ensure that the ship is maintained in conformity with relevant rules and regulations.
    10.2 The Company shall identify equipment that may cause hazardous situations if sudden failure occurs.

11. Documentation
    11.1 The Company shall establish and maintain procedures for controlling all documents and data relevant to the SMS.
    11.2 Documents shall be reviewed for adequacy and revised as necessary.

12. Company Verification, Review and Evaluation
    12.1 The Company shall conduct internal audits to verify whether activities comply with the SMS.
    12.2 The Company shall periodically evaluate the effectiveness of the SMS.

SECTION B - CERTIFICATION AND VERIFICATION
13. Certification
    13.1 The ship shall be operated by a Company holding a Document of Compliance (DOC).
    13.2 A Safety Management Certificate (SMC) shall be issued to each ship.
    13.3 The DOC and SMC shall be valid for a period of 5 years.

14. Interim Certification
    14.1 Interim DOC and interim SMC may be issued for new companies or new ships.

15. Verification
    15.1 The Administration shall verify compliance with the requirements of the ISM Code.
    15.2 Intermediate verification shall take place between the second and third anniversary date.

16. Forms of Certificates
    16.1 The DOC and SMC shall be drawn up in the form prescribed by the Administration.
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved ISM Code ({len(text)} chars)")
    return text


def collect_mlc():
    out = RAW_DIR / "mlc_2006.txt"
    if out.exists():
        return out.read_text("utf-8")
    text = """MARITIME LABOUR CONVENTION, 2006 (MLC 2006) - KEY PROVISIONS

TITLE 1 - MINIMUM REQUIREMENTS FOR SEAFARERS TO WORK ON A SHIP
Regulation 1.1 - Minimum Age
No person under the age of 16 shall be employed on a ship.
No person under the age of 18 shall be employed at night.
Night work for seafarers under 18 shall be prohibited.

Regulation 1.2 - Medical Certificate
Seafarers shall hold a valid medical certificate before commencing work.
Medical certificates shall be issued by a qualified medical practitioner.
Medical certificates shall be valid for a maximum period of two years.

Regulation 1.3 - Training and Qualifications
Seafarers shall be trained and qualified to perform their duties.
Seafarers shall hold a certificate of competency as required by STCW.

Regulation 1.4 - Recruitment and Placement
Seafarers shall have access to an efficient, adequate and accountable system for finding employment.
Recruitment and placement services shall be regulated by the competent authority.

TITLE 2 - CONDITIONS OF EMPLOYMENT
Regulation 2.1 - Seafarers' Employment Agreements
Every seafarer shall have a seafarers' employment agreement.
The agreement shall be signed by both the seafarer and the shipowner.
The seafarer shall be given an opportunity to review the agreement before signing.

Regulation 2.2 - Wages
Seafarers shall be paid for their work at regular intervals.
All seafarers shall receive a monthly account of payments.
Seafarers shall be allotted part of their wages to their families.

Regulation 2.3 - Hours of Work and Rest
Hours of work shall not exceed 14 hours in any 24-hour period.
Hours of rest shall not be less than 10 hours in any 24-hour period.
Hours of rest shall not be less than 77 hours in any 7-day period.

Regulation 2.4 - Entitlement to Leave
Seafarers shall be entitled to annual leave with pay.
Annual leave shall be calculated on the basis of a minimum of 2.5 calendar days per month of employment.

Regulation 2.5 - Repatriation
Seafarers shall be entitled to repatriation at no cost to themselves.
Circumstances giving right to repatriation include: expiry of the agreement; termination of agreement by the shipowner; illness or injury; and shipwreck.

Regulation 2.6 - Seafarer Compensation for Ship's Loss or Foundering
Seafarers shall be compensated for injury, loss or unemployment arising from shipwreck.

Regulation 2.7 - Manning Levels
Ships shall be sufficiently and safely manned.
Manning levels shall be adequate in terms of number and qualifications.

Regulation 2.8 - Career and Skill Development
Every seafarer shall have access to career and skill development opportunities.

TITLE 3 - ACCOMMODATION, RECREATIONAL FACILITIES, FOOD AND CATERING
Regulation 3.1 - Accommodation and Recreational Facilities
Each seafarer shall have a separate cabin.
Cabin floor area shall not be less than 4.5 square meters per person.
Cabins shall be provided with adequate lighting, ventilation and heating.
Sanitary facilities shall be provided for all seafarers.

Regulation 3.2 - Food and Catering
Food shall be provided free of charge to seafarers.
Food shall be of appropriate nutritional value and quality.
Catering staff shall be trained and qualified.

TITLE 4 - HEALTH PROTECTION, MEDICAL CARE, WELFARE AND SOCIAL SECURITY PROTECTION
Regulation 4.1 - Medical Care on Board Ship and Ashore
Seafarers shall have access to prompt and adequate medical care while working on board.
Ships shall carry a medicine chest and medical equipment.
Ships of 500 gross tonnage and above shall have a qualified medical doctor on board.

Regulation 4.2 - Shipowners' Liability
Shipowners shall provide health protection and medical care for seafarers.
Shipowners shall bear the cost of medical care and board and lodging for sick seafarers.

Regulation 4.3 - Health and Safety Protection and Accident Prevention
Safe and hygienic conditions shall be provided on ships.
Occupational safety and health policies shall be established.

Regulation 4.4 - Access to Shore-based Welfare Facilities
Seafarers shall have access to shore-based welfare facilities.
Welfare facilities shall be provided in all ports.

Regulation 4.5 - Social Security
Seafarers shall be provided with social security protection.
Protection shall include medical care, sickness benefit, and employment injury benefit.

TITLE 5 - COMPLIANCE AND ENFORCEMENT
Regulation 5.1 - Flag State Responsibilities
Each flag State shall implement and enforce the requirements of the Convention.
Ships shall carry a Maritime Labour Certificate and a Declaration of Maritime Labour Compliance.

Regulation 5.1.1 - Maritime Labour Certificate
The Maritime Labour Certificate shall be valid for a maximum of 5 years.
An intermediate inspection shall be carried out.

Regulation 5.1.2 - Declaration of Maritime Labour Compliance
The DMLC shall contain information on working and living conditions.
The DMLC shall be attached to the Maritime Labour Certificate.

Regulation 5.1.3 - Inspections
Flag States shall carry out inspections to verify compliance.
Inspections shall be carried out at intervals not exceeding 3 years.

Regulation 5.2 - Port State Responsibilities
Port State Control officers may inspect foreign ships to verify compliance.
Port State Control shall include verification of the Maritime Labour Certificate.

Regulation 5.2.1 - Inspections in Port
A Port State Control inspection may be carried out when there are clear grounds for believing that the ship does not conform.

Regulation 5.2.2 - Detailed Inspection
A detailed inspection may be carried out in case of suspected deficiencies.

Regulation 5.3 - Labour Supplying Responsibilities
Labour supplying countries shall regulate seafarer recruitment and placement services.
"""
    out.write_text(text, "utf-8")
    print(f"[collector] Saved MLC 2006 ({len(text)} chars)")
    return text


def collect_all():
    texts = {}
    collectors = [
        ("海商法", collect_maritime_code),
        ("海上交通安全法", collect_maritime_traffic_safety_law),
        ("海洋环境保护法", collect_marine_environment_law),
        ("船舶登记条例", collect_ship_regulations),
        ("船员条例", collect_crew_regulations),
        ("船舶吨税法", collect_ship_tonnage_tax_law),
        ("国际海运条例", collect_international_shipping_regulations),
        ("港口法", collect_port_law),
        ("内河交通安全管理条例", collect_inland_waterway_safety),
        ("防治船舶污染海洋环境管理条例", collect_ship_pollution_prevention),
        ("IMO Convention - SOLAS 详细", collect_detailed_solas),
        ("IMO Convention - MARPOL 详细", collect_detailed_marpol),
        ("IMO Convention - STCW 详细", collect_detailed_stcw),
        ("ISM Code (国际安全管理规则)", collect_ism_code),
        ("MLC 2006 (海事劳工公约)", collect_mlc),
    ]
    for name, func in collectors:
        try:
            text = func()
            if text:
                texts[name] = text
        except Exception as e:
            print(f"[collector] Failed to fetch {name}: {e}")
    return texts
