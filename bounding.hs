import Prelude
import Data.Functor

type Box = (Int, Int, Int, Int)

--area1 = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
area :: Box -> Int
area (left, top, right, bottom) = abs((right-left)*(bottom-top))

--x_overlap = Math.max(0, Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
xOverlap :: Box -> Box -> Int
xOverlap (left, top, right, bottom) (left', top', right', bottom') = max 0 ((min right right')-(max left left'))

--y_overlap = Math.max(0, Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
yOverlap :: Box -> Box -> Int
yOverlap (left, top, right, bottom) (left', top', right', bottom') = max 0 ((min bottom bottom')-(max top top'))

--intersection = x_overlap * y_overlap;
intersection :: Box -> Box -> Int
intersection b1 b2 = (xOverlap b1 b2) * (yOverlap b1 b2)

--union = area1 + area2 - intersection
union :: Box -> Box -> Int
union b1 b2 = (area b1) + (area b2) - (intersection b1 b2)

--iou = intersection/union
iou :: Box -> Box -> Float
iou b1 b2 = (fromIntegral(intersection b1 b2)) / (fromIntegral(union b1 b2))

--match = (iou >= 0.5)
match :: Box -> Box -> Bool
match b1 b2 = (iou b1 b2) >= 0.5
-----------------------------------------------------------------------------------------------------------------------
matchMany' :: [Box] -> Box -> [Bool]
matchMany' [] ground = []
matchMany' (x:xs) ground = match x ground : matchMany' xs ground

matchMany :: [Box] -> Box -> Bool
matchMany detecteds ground = or (matchMany' detecteds ground)

matchAll' :: [Box] -> [Box] -> [Bool]
matchAll' detecteds [] = []
matchAll' detecteds (x:xs) = matchMany detecteds x : matchAll' detecteds xs

--this function has the potential to incorrectly match the same detected box to multiple ground boxes if the ground boxes are very close together
--this could therefore give a higher fscore than should be given but it shouldn't happen in practice with any of the test images
matchAll :: [Box] -> [Box] -> Int
matchAll detecteds grounds = sum(fmap (\x -> if x == True then 1 else 0) (matchAll' detecteds grounds))
-----------------------------------------------------------------------------------------------------------------------
getF :: [Box] -> [Box] -> Float --get F, given ground boxes and detected boxes
getF gs ds = fromIntegral (matchAll ds gs)

getDA :: [Box] -> Float --get D, given detected boxes or A, given ground boxes
getDA das = fromIntegral (length das)
-----------------------------------------------------------------------------------------------------------------------
fScore :: [Box] -> [Box] -> Float
fScore gs ds = 2*f/(d+a) where
  f = getF gs ds
  d = getDA ds
  a = getDA gs

tpr :: [Box] -> [Box] -> Float
tpr gs ds = f/a where
  f = getF gs ds
  a = getDA gs

test :: [Box] -> [Box] -> IO Float
test gs ds = do
  putStrLn ("F-Score: " ++ show (fScore gs ds) ++ ", TPR: " ++ show (100*(tpr gs ds)) ++ "%\n")
  return (fScore gs ds)
-----------------------------------------------------------------------------------------------------------------------
parse :: String -> [[Box]]
parse s = boxed <$> (nums (words <$> (lines s)))
    where nums [] = []
          nums (x:xs) = (read <$> x) : (nums xs)
          boxed [] = []
          boxed (a:b:c:d:xs) = (a,b,c,d):(boxed xs)

testAll' :: [[Box]] -> [[Box]] -> Int -> Float -> IO()
testAll' [] [] n su = do
  putStrLn ("Average F-Score: " ++ show(su/16) ++ "\n")
  return ()
testAll' (x:xs) (y:ys) n su = do
  putStrLn (" Dart" ++ show(n) ++":")
  x <- test x y
  testAll' xs ys (n+1) (su + x)

testAll :: FilePath -> IO()
testAll file = do
  x <- readFile file
  testAll' allgrounds (parse x) 0 0
-----------------------------------------------------------------------------------------------------------------------
g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15 :: [Box]
g0 = [(423, 1, 620, 218)]
g1 = [(166, 106, 420, 356)]
g2 = [(88, 88, 201, 196),(317, 50, 378, 102)]
g3 = [(311, 138, 400, 233)]
g4 = [(157, 68, 417, 321)]
g5 = [(417, 126, 550, 252)]
g6 = [(204, 108, 281, 190)]
g7 = [(235, 155, 415, 334)]
g8 = [(824, 203, 975, 355),(63, 243, 134, 351)]
g9 = [(164, 17, 465, 318),(141, 532, 225, 587)]
g10 = [(77, 90, 99, 230),(577, 119, 646, 224),(912, 142, 957, 224)]
g11 = [(163, 94, 243, 187),(436, 104, 498, 192)]
g12 = [(147, 59, 227, 236)]
g13 = [(252, 101, 423, 271)]
g14 = [(102, 87, 265, 247),(968, 79, 1131, 240)]
g15 = [(130, 32, 304, 216)]

allgrounds :: [[Box]]
allgrounds = [g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15]
