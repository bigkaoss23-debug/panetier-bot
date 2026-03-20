// Service Claude API — structuration données factures (texte OCR) + NLP commandes
'use strict';

const Anthropic = require('@anthropic-ai/sdk');
const { TIMEOUTS, normalizeFournisseur } = require('../config');
const { withTimeout } = require('../utils/retry');

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const MODEL = 'claude-sonnet-4-6';

/**
 * Flusso 1 — Interprète le texte OCR d'une facture fournisseur.
 * Règle PRIX_UNITE: Montant ÷ Quantité livrée dans l'unité de vente réelle.
 * La "particularité" de la facture détermine l'unité (ex: "4x250g" → unité = sachet, qté = nb sachets).
 */
async function analyzeInvoiceOcrText(ocrText, isHandwritten = false) {
  if (!ocrText || ocrText.trim().length === 0)
    throw new Error('Texte OCR vide — aucun texte détecté dans l\'image');

  const handwrittenNote = isHandwritten
    ? '\nATTENTION: Texte manuscrit — peut contenir des fautes OCR. Interprète au mieux.'
    : '';

  const systemPrompt = `Tu es un assistant expert en gestion de stocks pour une boulangerie-pâtisserie suisse.
Tu reçois du texte brut extrait par OCR depuis des factures fournisseurs suisses.
Tu structures ces données en JSON propre et utilisable.
Tu réponds UNIQUEMENT en JSON valide, sans texte supplémentaire, sans balises markdown.${handwrittenNote}`;

  const userPrompt = `Voici le texte extrait par OCR depuis une facture fournisseur:

---
${ocrText}
---

Identifie et extrais TOUS les produits présents, MÊME si le prix est 0.

ÉTAPE 0 — Cherche la date de la facture (date du document, pas aujourd'hui).
Elle se trouve généralement en haut: "Date: 10.03.2026", "Genève le 10.03.2026", "18/03/2026", etc.
Normalise en DD.MM.YYYY. Si absente, utilise null.

ÉTAPE 0B — Cherche le TOTAL FINAL TVA INCLUSE de la facture.
Cherche la ligne "Total Bulletin de livraison" ou "Total de la livraison" tout en bas.
Cette ligne contient PLUSIEURS montants sur la même ligne. Ex: "29.34 | 661.84 | CHF 691.20"
RÈGLE: prends TOUJOURS le DERNIER montant de cette ligne (le plus à droite) = TVA incl.
Dans l'exemple: 691.20 ✅ (PAS 661.84 ❌, PAS 29.34 ❌)
Le montant TVA incl. est souvent précédé de "CHF" et est le plus grand de la ligne.
Même valeur pour tous les produits. Si absent, utilise null.

Pour chaque produit, retourne un objet JSON avec ces champs:
- dateFattura: date de la facture DD.MM.YYYY (string, même valeur pour tous, null si absente)
- totalFattura: total HT final de la facture (number, même valeur pour tous, null si absent)
- produit: nom commercial en Title Case, sans codes ni provenance
- fournisseur: Transgourmet, Léguriviera, DelMaitre, Fromages Chaudron, Volailles Importation, Maître Boucher, ou Inconnu
- montant: montant HT total de la ligne (number, 0 si non livré)
- quantiteLivree: quantité de contenants livrés (number)
- unite: unité pour comparer les prix — "lt" pour liquides, "kg" pour viandes/fromages, "pce" pour pièces/boîtes
- quantiteUnites: nombre d'unités dans UN contenant. Depuis colonne Poids/contenu: "20 lt"→20, "6 bo"→6, "12 pc"→12. Si absent→1.
- prixContenant: montant HT d'UN contenant = montant ÷ quantiteLivree (2 décimales)
- prixUnite: prix par unité interne = prixContenant ÷ quantiteUnites. Si colonne Prix présente → utilise-la directement. (0 si montant=0)
- particularite: colonne Particularité si présente (null sinon)

EXEMPLES TRANSGOURMET:
Economy Huile 20l (bx, 20lt, Prix=4.49, Montant=89.80, Livré=1): unite=lt, quantiteUnites=20, prixContenant=89.80, prixUnite=4.49
Oatly Barista 6x1l (ct, 6tp, Prix=3.42, Montant=41.04, Livré=2): unite=pce, quantiteUnites=6, prixContenant=20.52, prixUnite=3.42
Emmi Crème 35% 1l (pc, Prix=6.65, Montant=19.95, Livré=3): unite=lt, quantiteUnites=1, prixContenant=6.65, prixUnite=6.65
MSC Thon 6x1.88kg (ct, 6bo, Prix=20.28, Montant=121.68, Livré=1): unite=pce, quantiteUnites=6, prixContenant=121.68, prixUnite=20.28
Suma Maxi 20l (bi, Prix=124.85, Montant=124.85, Livré=1): unite=lt, quantiteUnites=20, prixContenant=124.85, prixUnite=6.24

EXEMPLES FROMAGES CHAUDRON (colonne Prix de vente):
Canelé Vanille SG 75x60g (Livré=2, Prix de vente=46.78, Montant=93.56): unite=pce, quantiteUnites=2, prixContenant=46.78, prixUnite=46.78
Pain Raisins 40x130g (Livré=12, Prix de vente=60.80, Montant=2054.56): unite=pce, quantiteUnites=12, prixContenant=171.21, prixUnite=60.80

EXEMPLES DELMAITRE (colonne Poids Net en kg):
Roastbeef 300g (Livré=6, Poids=1.760kg, Prix=38.20, Montant=67.23): unite=kg, quantiteUnites=1.760, prixContenant=11.21, prixUnite=38.20
Salami Nostrano (Livré=9, Poids=2.708kg, Prix=19.30, Montant=52.26): unite=kg, quantiteUnites=2.708, prixContenant=5.81, prixUnite=19.30

RÈGLES:
- INCLURE tous les produits même montant=0
- Ignorer totaux, TVA, frais de port, emballages IFCO
- Jamais utiliser prix TTC
- RÈGLE ABSOLUE: lire unite et quantiteUnites UNIQUEMENT depuis la colonne Poids/contenu de la facture
- prixUnite = colonne Prix de la facture si présente. Sinon = prixContenant
- JAMAIS lire grammages ou chiffres dans le nom du produit (750ml, 200x12g, 6x1l, 965g...)
- Si Poids/contenu ABSENT ou VIDE: unite = valeur de la colonne Un. (bx, ct, bi, bt, pc...), quantiteUnites=1, prixUnite=prixContenant
- DÉCIMALES OBLIGATOIRES: tous les nombres doivent avoir 2 décimales exactes (10.94 pas 10, 5.47 pas 5, 0.82 pas 0)
- Les factures suisses utilisent la virgule comme séparateur décimal — convertis toujours en point pour le JSON (ex: 10,94 → 10.94)

Réponds UNIQUEMENT avec le tableau JSON, sans texte avant ou après:
[{"dateFattura":"18.03.2026","totalFattura":703.44,"produit":"...","fournisseur":"...","montant":10.94,"quantiteLivree":2,"unite":"bq","quantiteUnites":1,"prixContenant":5.47,"prixUnite":5.47,"particularite":null}]`;

  const message = await withTimeout(
    () => client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    }),
    TIMEOUTS.CLAUDE_API,
    'Claude.analyzeInvoice'
  );

  const rawText = message.content[0]?.text?.trim() || '[]';

  try {
    // Extraction robuste: cherche le tableau JSON même si Sonnet écrit du texte avant
    let jsonStr = rawText.replace(/```json|```/g, '').trim();
    const startIdx = jsonStr.indexOf('[');
    const endIdx = jsonStr.lastIndexOf(']');
    if (startIdx >= 0 && endIdx > startIdx) {
      jsonStr = jsonStr.substring(startIdx, endIdx + 1);
    }
    const parsed = JSON.parse(jsonStr);
    const items = Array.isArray(parsed) ? parsed : [parsed];
    return items.map(p => ({ ...p, fournisseur: normalizeFournisseur(p.fournisseur) }));
  } catch (err) {
    console.error('❌ Erreur parsing JSON Claude (facture OCR):', rawText);
    throw new Error('Claude n\'a pas retourné un JSON valide pour la facture');
  }
}


async function parseOrderText(text, senderName) {
  const now = new Date();
  const dd = String(now.getDate()).padStart(2, '0');
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const yyyy = now.getFullYear();
  const today = `${dd}.${mm}.${yyyy}`;

  const systemPrompt = `Tu es un assistant pour une boulangerie-pâtisserie suisse nommée Le Panetier Sion.
Tu interprètes des messages de commande client et extrais les informations structurées.
Tu réponds UNIQUEMENT en JSON valide, sans texte supplémentaire, sans balises markdown.
Aujourd'hui nous sommes le ${today}.`;

  const userPrompt = `Interprète ce message de commande et retourne un JSON avec ces champs:
- nomClient: prénom et nom du client (string, ou "${senderName}" si non mentionné)
- produits: nom(s) du/des produit(s) commandé(s) (string)
- quantite: quantité et unité, ex: "2x", "500g", "1 pièce" (string)
- dateRetrait: date de retrait au format DD.MM.YYYY (string, ou null si non précisée)
- heureRetrait: heure de retrait au format HH:MM (string, ou null si non précisée)
- notes: remarques spéciales, allergènes, demandes particulières (string, ou null)
- telephone: numéro de téléphone si mentionné (string, ou null)

Message à analyser:
"${text}"

Retourne uniquement le JSON: {"nomClient":"...","produits":"...","quantite":"...","dateRetrait":"...","heureRetrait":"...","notes":null,"telephone":null}`;

  const message = await withTimeout(
    () => client.messages.create({
      model: MODEL,
      max_tokens: 512,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    }),
    TIMEOUTS.CLAUDE_API,
    'Claude.parseOrder'
  );

  const rawText = message.content[0]?.text?.trim() || '{}';

  try {
    const cleaned = rawText.replace(/```json|```/g, '').trim();
    return JSON.parse(cleaned);
  } catch (err) {
    console.error('❌ Erreur parsing JSON Claude (commande):', rawText);
    throw new Error('Claude n\'a pas retourné un JSON valide pour la commande');
  }
}


async function classifyDocument(ocrText, isHandwritten) {
  const systemPrompt = `Tu es un expert en classification de documents pour une boulangerie-pâtisserie suisse.
Tu analyses le texte OCR d'un document et tu identifies son type.
Tu réponds UNIQUEMENT avec un seul mot parmi: FACTURE, AGENDA, ACHAT`;

  const userPrompt = `Texte OCR (manuscrit: ${isHandwritten}):
---
${ocrText.substring(0, 800)}
---

RÈGLES DE CLASSIFICATION — dans cet ordre de priorité:

0. ACHAT en PRIORITÉ si le texte contient UN de ces éléments:
   - Mots clés: "acheter", "achater", "commander", "manque", "besoin", "à acheter", "achat"
   - Liste simple: quantité + ingrédient (ex: "5kg tomate", "3L lait") SANS prix et SANS totaux
   → Si présent → ACHAT immédiatement, sans vérifier les autres règles

1. FACTURE si le texte contient AU MOINS 2 de ces éléments:
   - Prix en CHF avec décimales (ex: "15.00", "8.42", "44.27")
   - Mots: "Total", "TVA", "TTC", "Bulletin de livraison", "Montant"
   - Nom de fournisseur connu: Transgourmet, Léguriviera, Chaudron, Maître Boucher
   - Structure tableau avec colonnes quantité/prix

2. AGENDA si le texte contient:
   - Texte manuscrit avec noms de clients (Mme, M., prénom+nom)
   - Produits de boulangerie (croissants, tarte, entremets, pain...)
   - Heures de retrait écrites à la main (07h30, 8h00...)
   - Numéros de téléphone
   NB: Les jours imprimés (März, Vendredi...) seuls ne suffisent PAS — il faut aussi du texte manuscrit

3. ACHAT si le texte contient:
   - Liste d'ingrédients à acheter avec quantités
   - SANS noms de clients, SANS horaires de retrait

Réponds UNIQUEMENT avec: FACTURE, AGENDA, ou ACHAT`;

  const message = await withTimeout(
    () => client.messages.create({
      model: MODEL,
      max_tokens: 10,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    }),
    TIMEOUTS.CLAUDE_API,
    'Claude.classifyDocument'
  );

  const result = message.content[0]?.text?.trim().toUpperCase();
  if (['FACTURE', 'AGENDA', 'ACHAT'].includes(result)) return result;
  return 'INCONNU';
}


async function analyzeAgendaDocument(ocrText, pageDate = null) {
  if (!ocrText || ocrText.trim().length === 0)
    throw new Error('Texte OCR vide');

  const now = new Date();
  const yyyy = now.getFullYear();
  const dd = String(now.getDate()).padStart(2, '0');
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const today = `${dd}.${mm}.${yyyy}`;

  const systemPrompt = `Tu es un expert en analyse de documents pour une exploitation de restauration et boulangerie/pâtisserie. Ta mission est d'identifier le type de document et d'extraire les données au format JSON.

ÉTAPE 1 - TRANSCRIPTION BRUTE :
Lis et transcris TOUT le texte extrait par OCR, mot par mot. Ignore toute structure d'agenda (créneaux horaires, lignes, colonnes). Traite le contenu comme une feuille de notes ordinaire.

ÉTAPE 1.5 - CORRECTION ET INTERPRÉTATION :
Le texte OCR peut contenir des erreurs de lecture. Nous sommes dans une boulangerie/pâtisserie. Applique ces règles :
- Un nombre suivi d'un mot est toujours un PRODUIT avec quantité
- "ao Beuille" ou "ao Beurle" = "au beurre"
- "k" après un nombre = "x" (quantité): "20k" → "20x"
- Ne jamais confondre un produit avec un nom de client

ÉTAPE 2 - INTERPRÉTATION : identifie noms clients, produits, horaires, téléphones.

RÈGLES CRITIQUES :
- Les symboles ❓ et ? ne doivent JAMAIS apparaître dans les champs JSON — si un produit est inconnu ou difficile à lire, garde le texte exact tel que lu par l'OCR (ex: "Tradilins", "Trdlins"...), ne le remplace JAMAIS par "?" ou "❓"
- Les chiffres sont lus EXACTEMENT tels qu'écrits — ne jamais réinterpréter un chiffre (4 reste 4, jamais 2 ou autre)
- La date est extraite de l'en-tête de la page d'agenda (ex: "13 März/Mars/Marzo" → "13.03.${yyyy}")
- Aujourd'hui: ${today}

ÉTAPE 3 - CLASSIFICATION :
- COMMANDES_CLIENTS : commande avec produit + horaire + nom client
- ACHATS_LOGISTIQUE : liste d'ingrédients SANS nom de client et SANS horaire

STRUCTURE AGENDA SUISSE:
- Les agendas ont des créneaux horaires à gauche (7.00, 7.30, 8.00...)
- La commande est écrite sur la ligne de l'heure de retrait
- La ligne suivante contient souvent le nom du client et son téléphone
- Une ligne horizontale = séparation entre deux clients
- Si COMMANDES_CLIENTS ET horaire présent → TOUJOURS COMMANDES_CLIENTS

RÈGLE D'OR : Un objet JSON = Un client unique

Tu réponds UNIQUEMENT avec le JSON pur, sans texte avant ou après.`;

  const userPrompt = `Texte extrait de l'image:
---
${ocrText}
---

Retourne UNIQUEMENT ce JSON:
{"commandes": [{"flux": "COMMANDES_CLIENTS ou ACHATS_LOGISTIQUE","contact": "Nom du client","produit": "Liste des articles","quantite": "Quantités","Date_Retrait": "DD.MM.YYYY","heure_retrait": "HH:MM","telephone": "Numéro","notes": "Détails"}]}`;

  const message = await withTimeout(
    () => client.messages.create({
      model: MODEL,
      max_tokens: 1024,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    }),
    TIMEOUTS.CLAUDE_API,
    'Claude.analyzeAgenda'
  );

  const rawText = message.content[0]?.text?.trim() || '{"commandes":[]}';
  console.log('🤖 Claude AGENDA raw output:', rawText);
  try {
    const cleaned = rawText.replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    console.log('📋 Claude AGENDA parsed:', JSON.stringify(parsed, null, 2));
    return Array.isArray(parsed.commandes) ? parsed.commandes : [];
  } catch (err) {
    console.error('❌ Erreur parsing JSON agenda:', rawText);
    throw new Error("Claude n'a pas retourné un JSON valide pour l'agenda");
  }
}


/**
 * Flusso ACHAT — Interprète un message texte comme une liste d'achats.
 * Retourne un tableau d'achats avec produit, quantite, unite, dateLivraison, notes.
 */
async function parseAchatText(text, senderName) {
  const now = new Date();
  const dd = String(now.getDate()).padStart(2, '0');
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const yyyy = now.getFullYear();
  const today = `${dd}.${mm}.${yyyy}`;

  // Calcul demain
  const tom = new Date(now.getTime() + 86400000);
  const demain = `${String(tom.getDate()).padStart(2,'0')}.${String(tom.getMonth()+1).padStart(2,'0')}.${tom.getFullYear()}`;

  const systemPrompt = `Tu es un assistant pour une boulangerie-pâtisserie suisse.
Tu interprètes des messages de demande d'achat d'ingrédients.
Tu réponds UNIQUEMENT en JSON valide, sans texte avant ou après.
Aujourd'hui: ${today} | Demain: ${demain}`;

  const userPrompt = `Message: "${text}"
Demandeur: ${senderName}

Extrais les achats demandés. Pour chaque produit:
- produit: nom de l'ingrédient (string)
- quantite: nombre (number)
- unite: kg/pce/l/sachet/etc (string)
- dateLivraison: DD.MM.YYYY (string, null si non précisé)
- notes: remarques (string, null)

"x demain" ou "pour demain" → dateLivraison = ${demain}
"urgent" ou "aujourd'hui" → dateLivraison = ${today}

Retourne UNIQUEMENT ce JSON:
{"achats": [{"produit":"...","quantite":0,"unite":"kg","dateLivraison":"DD.MM.YYYY","notes":null}]}`;

  const message = await withTimeout(
    () => client.messages.create({
      model: MODEL,
      max_tokens: 512,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }]
    }),
    TIMEOUTS.CLAUDE_API,
    'Claude.parseAchat'
  );

  const rawText = message.content[0]?.text?.trim() || '{"achats":[]}';
  try {
    const cleaned = rawText.replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    return Array.isArray(parsed.achats) ? parsed.achats : [];
  } catch (err) {
    console.error('Erreur parsing JSON achat:', rawText);
    throw new Error("Claude n'a pas retourné un JSON valide pour l'achat");
  }
}

module.exports = { analyzeInvoiceOcrText, parseOrderText, classifyDocument, analyzeAgendaDocument, parseAchatText };
