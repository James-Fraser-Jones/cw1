import Prelude

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

--uoi = union/intersection
iou :: Box -> Box -> Float
iou b1 b2 = (fromIntegral(intersection b1 b2)) / (fromIntegral(union b1 b2))

--match = (uoi >= 0.5)
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

matchAll :: [Box] -> [Box] -> Int
matchAll detecteds grounds = sum(fmap (\x -> if x == True then 1 else 0) (matchAll' detecteds grounds))
-----------------------------------------------------------------------------------------------------------------------
ground0 = (423,1,620,218)
