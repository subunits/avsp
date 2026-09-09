-- Augmented Vector Space Playground v3
-- No dependencies — runs on play.haskell.org (base only)
-- Compile: ghc -O Main.hs -o avs
-- Run:     ./avs [input.csv]

module Main where

import Data.List  (sortBy, intercalate)
import Data.Ord   (comparing)
import Data.Char  (isSpace)

-- ─── Types ───────────────────────────────────────────────────────────────────

type Sample = Double
type Vec    = [Double]
type Label  = Int

data AugConfig = AugConfig
  { lagWindow  :: Int
  , derivOrder :: Int
  , rollingWin :: Int
  } deriving (Show)

data AugPoint = AugPoint
  { apLabel  :: Label
  , apVec    :: Vec
  , apSignal :: String
  } deriving (Show)

data Metric = L2 | Cosine deriving (Show, Eq)

-- ─── Signal generators ───────────────────────────────────────────────────────

sineWave :: Int -> [Sample]
sineWave n = [ sin (2 * pi * fromIntegral t / 20) * 3 | t <- [0..n-1] ]

stepWave :: Int -> [Sample]
stepWave n = zipWith (+) steps noise
  where
    steps = [ if t < 20 then 1 else if t < 40 then -1 else 2 | t <- [0..n-1] ]
    noise = take n (gaussStream 42)

logisticMap :: Int -> [Sample]
logisticMap n = take n . map (\x -> (x - 0.5) * 6) $ iterate step 0.5
  where step x = 3.9 * x * (1 - x)

lorenzWave :: Int -> [Sample]
lorenzWave n = normalise . take n . drop 50 $ map (\(x,_,_) -> x) lorenzStream
  where
    lorenzStream = iterate lorenzStep (1.0, 1.0, 1.0)
    lorenzStep (x,y,z) =
      let dt = 0.02; s = 10; rho = 28; beta = 8/3
      in ( x + dt * s   * (y - x)
         , y + dt * (x  * (rho - z) - y)
         , z + dt * (x  * y - beta * z) )
    normalise xs =
      let mx = maximum (map abs xs)
      in  map (* (3 / max mx 1e-9)) xs

sawtoothWave :: Int -> [Sample]
sawtoothWave n =
  [ let base  = fromIntegral (t `mod` 15) / 15 * 4 - 2
        noise = sin (fromIntegral t * 2.3) * 0.4
              + sin (fromIntegral t * 0.7) * 0.3
    in  base + noise
  | t <- [0..n-1] ]

randomWalk :: Int -> [Sample]
randomWalk n = take n $ scanl (+) 0 (map (* 0.8) (uniformStream 12345))

-- ─── Deterministic noise streams ─────────────────────────────────────────────

uniformStream :: Int -> [Double]
uniformStream seed = map toFloat (iterate lcg (fromIntegral seed))
  where
    lcg s   = (s * 1664525 + 1013904223) `mod` (2^32)
    toFloat s = fromIntegral s / fromIntegral (2^32 :: Integer) - 0.5

gaussStream :: Int -> [Double]
gaussStream seed =
  [ sqrt (-2 * log (max u 1e-9)) * cos (2 * pi * v) * 0.6
  | (u, v) <- pairs (map (+ 0.5) (uniformStream seed)) ]
  where
    pairs (a:b:rest) = (a,b) : pairs rest
    pairs _          = []

-- ─── Feature extractors ──────────────────────────────────────────────────────

lagFeats :: Int -> [Sample] -> [(Int, Vec)]
lagFeats p xs =
  [ (i, reverse (take p (drop (i - p + 1) xs)))
  | i <- [p-1 .. length xs - 1] ]

diffs :: Int -> [Sample] -> [Sample]
diffs 0 xs = xs
diffs d xs = diffs (d-1) (zipWith (-) (drop 1 xs) xs)

rollingStats :: Int -> [Sample] -> [(Double, Double)]
rollingStats w xs =
  [ let win = take w (drop i xs)
        mu  = sum win / fromIntegral w
        var = sum (map (\x -> (x - mu)^2) win) / fromIntegral w
    in (mu, sqrt var)
  | i <- [0 .. length xs - w] ]

-- ─── Augmentation ────────────────────────────────────────────────────────────

augment :: AugConfig -> String -> [Sample] -> [AugPoint]
augment cfg name xs = zipWith3 build lagPairs diffSeries statSeries
  where
    p = lagWindow  cfg
    d = derivOrder cfg
    w = rollingWin cfg
    lagPairs   = lagFeats p xs
    start      = p - 1
    diffXs     = diffs 1 xs
    diff2Xs    = diffs 2 xs
    diffSeries = case d of
      0 -> repeat []
      1 -> map (:[]) (drop start diffXs)
      _ -> zipWith (\a b -> [a,b]) (drop start diffXs) (drop start diff2Xs)
    rawStats   = rollingStats w xs
    statSeries = drop (start - (w-1)) rawStats
    build (idx, lags) ds (mu, sigma) =
      AugPoint idx (lags ++ ds ++ [mu, sigma]) name

-- ─── Distance functions ───────────────────────────────────────────────────────

l2 :: Vec -> Vec -> Double
l2 a b = sqrt . sum $ zipWith (\x y -> (x-y)^2) a b

cosine :: Vec -> Vec -> Double
cosine a b =
  let dot = sum (zipWith (*) a b)
      na  = sqrt (sum (map (^2) a))
      nb  = sqrt (sum (map (^2) b))
      den = na * nb
  in  if den < 1e-12 then 1.0 else 1.0 - dot / den

applyMetric :: Metric -> Vec -> Vec -> Double
applyMetric L2     = l2
applyMetric Cosine = cosine

-- ─── kNN ─────────────────────────────────────────────────────────────────────

knn :: Metric -> Int -> AugPoint -> [AugPoint] -> [(AugPoint, Double)]
knn metric k query corpus =
  take k
  . sortBy (comparing snd)
  . map (\p -> (p, applyMetric metric (apVec query) (apVec p)))
  . filter ((/= apLabel query) . apLabel)
  $ corpus

-- ─── Anomaly detection ───────────────────────────────────────────────────────

anomalyThreshold :: Double
anomalyThreshold = 2.0

detectAnomalies :: Metric -> Int -> [AugPoint] -> [(AugPoint, Double)]
detectAnomalies metric k pts =
  let scored     = map (\p -> (p, meanDist p)) pts
      meanDist p = let ns = knn metric k p pts
                   in  if null ns then 0
                       else sum (map snd ns) / fromIntegral (length ns)
      globalMean = sum (map snd scored) / fromIntegral (length scored)
      threshold  = anomalyThreshold * globalMean
  in  filter (\(_,d) -> d > threshold) scored

-- ─── Shannon entropy (binned) ────────────────────────────────────────────────

shannonEntropy :: [Double] -> Double
shannonEntropy [] = 0
shannonEntropy xs =
  let n    = length xs
      bins = max 4 (round (sqrt (fromIntegral n)) :: Int)
      mn   = minimum xs
      mx   = maximum xs
      rng  = max (mx - mn) 1e-9
      idx v = min (bins-1) (floor ((v - mn) / rng * fromIntegral bins) :: Int)
      counts = foldr (\v acc ->
                 let i = idx v
                     (pre, c:post) = splitAt i acc
                 in  pre ++ (c+1) : post
               ) (replicate bins 0) xs
      total = fromIntegral n :: Double
  in  negate . sum $
        [ let p = fromIntegral c / total
          in  if p > 0 then p * logBase 2 p else 0
        | c <- counts ]

-- ─── OLS regression (coordinate descent) ─────────────────────────────────────

dotV :: Vec -> Vec -> Double
dotV a b = sum (zipWith (*) a b)

cdStep :: Vec -> [(Vec, Double)] -> Vec
cdStep beta pairs =
  foldl updateCoord beta [0..length beta - 1]
  where
    updateCoord b i =
      let xi  = map (\(x,_) -> x !! i) pairs
          ri  = map (\(x,y) -> y - dotV b x + (b !! i) * (x !! i)) pairs
          num = dotV xi ri
          den = dotV xi xi
      in  if abs den < 1e-12 then b
          else take i b ++ [num / den] ++ drop (i+1) b

fitOLS :: Int -> [(Vec, Double)] -> Vec
fitOLS iters pairs =
  case pairs of
    []           -> []
    ((v,_) : _) -> iterate (`cdStep` pairs) (replicate (length v) 0.0) !! iters

predictOLS :: Vec -> Vec -> Double
predictOLS = dotV

regressionPairs :: [Sample] -> [AugPoint] -> [(Vec, Double)]
regressionPairs xs pts =
  [ (apVec p, xs !! (apLabel p + 1))
  | p <- pts
  , apLabel p + 1 < length xs ]

-- ─── Distance matrix ─────────────────────────────────────────────────────────

distMatrix :: Metric -> [AugPoint] -> [[Double]]
distMatrix metric pts =
  [ [ applyMetric metric (apVec a) (apVec b) | b <- pts ] | a <- pts ]

-- ─── Pretty printing ─────────────────────────────────────────────────────────

rule :: String
rule = "└" ++ replicate 66 '─' ++ "┘"

printSection :: String -> [String] -> IO ()
printSection hdr rows = do
  putStrLn ""
  putStrLn $ "┌─ " ++ hdr ++ " " ++ replicate (64 - length hdr) '─' ++ "┐"
  mapM_ (putStrLn . ("│  " ++)) rows
  putStrLn rule

bar :: Double -> Double -> Int -> String
bar v maxV w =
  let len = max 0 . min w $ round (v / max maxV 1e-9 * fromIntegral w)
  in replicate len '█' ++ replicate (w - len) '░'

fmtF :: Int -> Double -> String
fmtF dp x = show (fromIntegral (round (x * 10^dp)) / 10^dp :: Double)

padL :: Int -> String -> String
padL n s = replicate (max 0 (n - length s)) ' ' ++ s

padR :: Int -> String -> String
padR n s = take n (s ++ repeat ' ')

showVec :: Int -> Vec -> String
showVec n vs = "[" ++ intercalate ", " (map (padL 6 . fmtF 2) (take n vs))
            ++ (if length vs > n then ", …]" else "]")

-- ─── Reports ─────────────────────────────────────────────────────────────────

reportSignals :: AugConfig -> IO ()
reportSignals cfg = do
  let signals =
        [ ("sine",        sineWave     60)
        , ("logistic",    logisticMap  60)
        , ("lorenz",      lorenzWave   60)
        , ("sawtooth",    sawtoothWave 60)
        , ("step+noise",  stepWave     60)
        , ("random walk", randomWalk   60)
        ]
      w = 24
  printSection "Signals — norm per augmented point" $
    concatMap (\(name, xs) ->
      let pts  = augment cfg name xs
          ns   = map (\p -> (apLabel p, l2 (apVec p) (replicate (length (apVec p)) 0))) pts
          maxN = maximum (map snd ns)
          ent  = shannonEntropy xs
          rows = map (\(t,n) ->
                   "  " ++ padR 5 (show t)
                   ++ bar n maxN w
                   ++ "  " ++ fmtF 2 n) (take 10 ns)
      in  ("" : ("── " ++ name ++ "  (entropy=" ++ fmtF 2 ent ++ " bits)") : rows)
    ) signals

reportKNN :: [AugPoint] -> IO ()
reportKNN pts = do
  let queryT = 10
  case filter ((== queryT) . apLabel) pts of
    []        -> putStrLn "Query point not found."
    (query:_) -> do
      let runKNN metric = do
            let nbs  = knn metric 5 query pts
                maxD = maximum (map snd nbs)
            printSection ("5-NN of t=" ++ show queryT ++ " — " ++ show metric) $
              [ "Query vec : " ++ showVec 4 (apVec query) ++ " …"
              , "Signal    : " ++ apSignal query
              , "" ] ++
              map (\(p,d) ->
                "t=" ++ padR 3 (show (apLabel p))
                ++ " [" ++ padR 10 (apSignal p) ++ "]  "
                ++ bar d maxD 24
                ++ "  " ++ fmtF 4 d
                ) nbs
      runKNN L2
      runKNN Cosine

reportRegression :: [Sample] -> [AugPoint] -> IO ()
reportRegression xs pts = do
  let pairs  = regressionPairs xs pts
      beta   = fitOLS 200 pairs
      preds  = map (\(v,_) -> predictOLS beta v) pairs
      actual = map snd pairs
      errs   = zipWith (\p a -> p - a) preds actual
      mse    = sum (map (^2) errs) / fromIntegral (length errs)
      rmse   = sqrt mse
      maxE   = maximum (map abs errs)
  printSection "OLS regression — next-step prediction" $
    [ "Coefficients β : " ++ showVec 6 beta
    , ""
    , "RMSE           : " ++ fmtF 4 rmse
    , "Max |error|    : " ++ fmtF 4 maxE
    , ""
    , "  t      actual  predicted      error" ] ++
    zipWith3 (\p a e ->
      "  " ++ padR 4 (show (apLabel p))
      ++ padL 10 (fmtF 3 a)
      ++ padL 11 (fmtF 3 (predictOLS beta (apVec p)))
      ++ padL 11 (fmtF 4 e)
      ) (take 10 pts) (take 10 actual) (take 10 errs)

reportEntropy :: AugConfig -> [Sample] -> [AugPoint] -> IO ()
reportEntropy cfg xs pts = do
  case pts of
    [] -> return ()
    (p:_) ->
      let dim    = length (apVec p)
          cols   = [ map (\pt -> apVec pt !! i) pts | i <- [0..dim-1] ]
          ents   = map shannonEntropy cols
          maxE   = maximum ents
          labels = map (\i -> "dim " ++ show i) [0..dim-1]
      in  printSection "Shannon entropy per augmented dimension" $
            zipWith (\lbl e ->
              padR 8 lbl
              ++ bar e maxE 28
              ++ "  " ++ fmtF 3 e ++ " bits"
              ) labels ents

reportDistMatrix :: Metric -> [AugPoint] -> IO ()
reportDistMatrix metric pts = do
  let queries = filter (\p -> apLabel p `elem` [5,10,15,20,25,30]) pts
      mat     = distMatrix metric queries
      header  = "      " ++ concatMap (\p -> padL 7 ("t=" ++ show (apLabel p))) queries
      rows    = zipWith (\p row ->
                  padR 6 ("t=" ++ show (apLabel p))
                  ++ concatMap (\d -> padL 7 (fmtF 3 d)) row
                ) queries mat
  printSection ("Distance matrix — " ++ show metric
             ++ " (t=5,10,15,20,25,30)")
    (header : "" : rows)

reportAnomalies :: [AugPoint] -> IO ()
reportAnomalies pts = do
  let flagged = detectAnomalies L2 5 pts
  printSection ("Anomaly detection (threshold = "
             ++ show anomalyThreshold ++ "x global mean kNN distance)") $
    if null flagged
      then ["No anomalies detected."]
      else ["Flagged " ++ show (length flagged) ++ " point(s):", ""] ++
           map (\(p,d) ->
             "  t=" ++ padR 4 (show (apLabel p))
             ++ " [" ++ padR 10 (apSignal p) ++ "]"
             ++ "  mean-kNN-dist=" ++ fmtF 4 d
             ) flagged

-- ─── Main ────────────────────────────────────────────────────────────────────

defaultConfig :: AugConfig
defaultConfig = AugConfig { lagWindow = 3, derivOrder = 1, rollingWin = 4 }

main :: IO ()
main = do
  putStrLn "╔════════════════════════════════════════════════════════════════════╗"
  putStrLn "║     Augmented Vector Space Ground Tool  v3  ·  Haskell (base)    ║"
  putStrLn "╚════════════════════════════════════════════════════════════════════╝"

  -- Playground: no args, no file I/O — uses logistic map
  let sigName = "logistic"
      xs      = logisticMap 60
      cfg     = defaultConfig
      allPts  = concatMap (\(name, sig) -> augment cfg name sig)
                  [ ("sine",        sineWave     60)
                  , ("logistic",    logisticMap  60)
                  , ("lorenz",      lorenzWave   60)
                  , ("sawtooth",    sawtoothWave 60)
                  , ("step+noise",  stepWave     60)
                  , ("random walk", randomWalk   60)
                  ]
      sigPts  = augment cfg sigName xs

  reportSignals    cfg
  reportKNN        allPts
  reportRegression xs sigPts
  reportEntropy    cfg xs sigPts
  reportDistMatrix L2     sigPts
  reportDistMatrix Cosine sigPts
  reportAnomalies  sigPts

  putStrLn ""
  putStrLn "── Signal roster ───────────────────────────────────────────────────"
  putStrLn "   sineWave     — pure sine, period 20"
  putStrLn "   logisticMap  — deterministic chaos (r=3.9)"
  putStrLn "   lorenzWave   — Lorenz attractor x-component"
  putStrLn "   sawtoothWave — sawtooth + harmonic interference"
  putStrLn "   stepWave     — step + Gaussian noise"
  putStrLn "   randomWalk   — cumulative LCG noise"
